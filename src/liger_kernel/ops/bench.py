import torch
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.ops.experimental.orpo_loss import (
    odds_ratio_loss,
    LigerFusedLinearORPOFunction,
)

torch.set_default_device("cuda")


def f(m, batch, label, ignore_index=-100):
    weight = m.weight
    bias = m.bias

    len_chosen = batch.shape[0] // 2
    unnorm_logits = batch @ weight.t()  # chunk_size x V
    if bias is not None:
        unnorm_logits = unnorm_logits + bias
    concatenated_logits = F.log_softmax(unnorm_logits, dim=-1)
    chosen_nll_loss = F.nll_loss(
        concatenated_logits[:len_chosen].view(-1, concatenated_logits.shape[-1]),
        label[:len_chosen].view(-1),
        reduction="sum",
        ignore_index=ignore_index
    )

    all_logps = concatenated_logits.gather(-1, label.unsqueeze(2)).squeeze(2)
    chosen_logps = all_logps[:len_chosen].mean(dim=1)
    rejected_logps = all_logps[len_chosen:].mean(dim=1)

    or_loss = odds_ratio_loss(chosen_logps, rejected_logps)

    chosen_nll_loss /= (label[:len_chosen] != ignore_index).sum().item()
    or_loss /= batch.shape[0]

    loss = chosen_nll_loss + or_loss
    loss.backward()
    return loss


def liger_chunked_f(m, batch, label, compiled=True, pre_compiled=None):
    out = LigerFusedLinearORPOFunction.apply(batch, m.weight, label, m.bias, -100, compiled, pre_compiled)
    out.backward()
    return out


def bench(f, name=None, iters=100, warmup=5, display=True, profile=False, profile_mem=False):
    from triton.testing import do_bench

    for _ in range(warmup):
        f()

    if profile_mem:
        torch.cuda.memory._record_memory_history()
        f()
        torch.cuda.memory._dump_snapshot(f"{name if name is not None else 'memory'}.pickle")
    if profile:
        with torch.profiler.profile() as prof:
            f()
        prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")

    torch.cuda.reset_peak_memory_stats()
    ms_per_iter = do_bench(lambda: f())
    if name is None:
        res = ms_per_iter
    else:
        res = f"{name}: {ms_per_iter:.3f}ms"
    if display:
        print(res)
        print(f"Peak mem: {torch.cuda.max_memory_allocated()/1e9}gb")
        print()
    return res


B, T, D, V = 16, 1024, 768, 128256
model = nn.Linear(D, V).to(torch.bfloat16)
nll = nn.NLLLoss(reduction="sum")
ce = nn.CrossEntropyLoss()
chosen = torch.randn(B, T, D, requires_grad=True, dtype=torch.bfloat16)
rejected = torch.randn(B, T, D, requires_grad=True, dtype=torch.bfloat16)
chosen_label = torch.randint(0, V, (B, T)).to(torch.int64)
rejected_label = torch.randint(0, V, (B, T)).to(torch.int64)
concatenated_batch = torch.cat([chosen, rejected], dim=0)
concatenated_label = torch.cat([chosen_label, rejected_label], dim=0)


opt_f = torch.compile(f)
bench(lambda: f(model, concatenated_batch, concatenated_label), name="eager (ORPO non-chunked)")
bench(lambda: opt_f(model, concatenated_batch, concatenated_label), name="compile (ORPO non-chunked)")
bench(lambda: liger_chunked_f(model, concatenated_batch, concatenated_label, compiled=False, pre_compiled=None), name="eager (ORPO chunked)")
bench(lambda: liger_chunked_f(model, concatenated_batch, concatenated_label, compiled=True, pre_compiled=None), name="compile (ORPO chunked)")
bench(lambda: liger_chunked_f(model, concatenated_batch, concatenated_label, pre_compiled="original"), name="(pre)compile (ORPO chunked)")
bench(lambda: liger_chunked_f(model, concatenated_batch, concatenated_label, pre_compiled="modified"), name="compile + online softmax (ORPO chunked)")
