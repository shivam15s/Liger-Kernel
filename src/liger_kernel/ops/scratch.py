import torch
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.ops.experimental.orpo_loss import LigerFusedLinearORPOFunction

torch.set_default_device("cuda")


def odds_ratio_loss(chosen_logps, rejected_logps, beta=1.0):
    log_odds = (chosen_logps - rejected_logps) - (
        torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
    )
    sig_ratio = F.sigmoid(log_odds)
    ratio = torch.log(sig_ratio)
    losses = beta * ratio
    return losses.sum()


class ChunkedORPO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, target, bias=None, compiled=True):
        CHUNK_SIZE = 256

        def compute_orpo_loss(input_chunk, weight, bias, target):
            len_chosen_chunk = target.shape[0] // 2

            unnorm_logits = torch.addmm(bias, input_chunk, weight.t())
            unnorm_logits = unnorm_logits.float()
            norm_logits = F.log_softmax(unnorm_logits, dim=-1)

            chosen_nll_loss = nll(
                norm_logits[:len_chosen_chunk].view(-1, V),
                target[:len_chosen_chunk].view(-1),
            )
            all_logps = norm_logits.gather(-1, target.unsqueeze(1)).squeeze(1)
            chosen_logps = all_logps[:len_chosen_chunk]
            rejected_logps = all_logps[len_chosen_chunk:]

            or_loss = odds_ratio_loss(chosen_logps, rejected_logps)
            loss = chosen_nll_loss + or_loss
            return loss

        grad_weight = torch.zeros_like(weight)
        grad_chosen_inputs = []
        grad_rejected_inputs = []
        grad_bias = torch.zeros_like(bias)
        loss_acc = torch.zeros((), device=_input.device)

        chunks = _input.shape[0] // (2 * CHUNK_SIZE)

        def accumulate_chunk(input_chunk, target_chunk):
            (chunk_grad_input, chunk_grad_weight, chunk_grad_bias), chunk_loss = torch.func.grad_and_value(
                compute_orpo_loss, argnums=(0, 1, 2)
            )(input_chunk, weight, bias, target_chunk)
            grad_weight.add_(chunk_grad_weight)
            grad_bias.add_(chunk_grad_bias)
            loss_acc.add_(chunk_loss)
            return chunk_grad_input

        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)

        len_chosen = target.shape[0] // 2
        _chosen_input_chunks = torch.chunk(_input[:len_chosen], chunks=chunks, dim=0)
        _chosen_target_chunks = torch.chunk(target[:len_chosen], chunks=chunks, dim=0)
        _rejected_input_chunks = torch.chunk(_input[len_chosen:], chunks=chunks, dim=0)
        _rejected_target_chunks = torch.chunk(target[len_chosen:], chunks=chunks, dim=0)

        for (
            chosen_input_chunk,
            rejected_input_chunk,
            chosen_target_chunk,
            rejected_target_chunk,
        ) in zip(
            _chosen_input_chunks,
            _rejected_input_chunks,
            _chosen_target_chunks,
            _rejected_target_chunks,
        ):
            input_chunk = torch.cat([chosen_input_chunk, rejected_input_chunk], dim=0)
            target_chunk = torch.cat([chosen_target_chunk, rejected_target_chunk], dim=0)
            grad_input = accumulate_chunk(input_chunk, target_chunk)
            grad_chosen_inputs.append(grad_input[: chosen_target_chunk.shape[0]])
            grad_rejected_inputs.append(grad_input[chosen_target_chunk.shape[0] :])

        # combine grad_chosen_inputs and grad_rejected_inputs
        grad_inputs = grad_chosen_inputs + grad_rejected_inputs

        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0) / chunks,
            grad_weight / chunks,
            grad_bias / chunks,
        )
        return loss_acc

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        return grad_input, grad_weight, None, grad_bias, None


B, T, D, V = 32, 1024, 768, 128256
model = nn.Linear(D, V).to(torch.bfloat16)
nll = nn.NLLLoss(reduction="sum")
ce = nn.CrossEntropyLoss()
chosen = torch.randn(B, T, D, requires_grad=True, dtype=torch.bfloat16)
rejected = torch.randn(B, T, D, requires_grad=True, dtype=torch.bfloat16)
chosen_label = torch.randint(0, V, (B, T)).to(torch.int64)
rejected_label = torch.randint(0, V, (B, T)).to(torch.int64)
concatenated_batch = torch.cat([chosen, rejected], dim=0)
concatenated_label = torch.cat([chosen_label, rejected_label], dim=0)


def f(m, batch, label):
    len_chosen = batch.shape[0] // 2
    concatenated_logits = F.log_softmax(m(batch), dim=-1)
    chosen_nll_loss = nll(concatenated_logits[:len_chosen].view(-1, V), label[:len_chosen].view(-1))

    all_logps = concatenated_logits.gather(-1, label.unsqueeze(2)).squeeze(2)
    chosen_logps = all_logps[:len_chosen]
    rejected_logps = all_logps[len_chosen:]

    or_loss = odds_ratio_loss(chosen_logps, rejected_logps)

    loss = chosen_nll_loss + or_loss
    loss.backward()
    return loss


def chunked_f(m, batch, label, compiled=True):
    out = ChunkedORPO.apply(batch.view(-1, D), m.weight, label.view(-1), m.bias, compiled)
    out.backward()
    return out


def liger_chunked_f(m, batch, label, compiled=True):
    out = LigerFusedLinearORPOFunction.apply(batch, m.weight, label, m.bias, -100, compiled)
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


opt_f = torch.compile(f)
bench(lambda: f(model, concatenated_batch, concatenated_label), name="eager (non-chunked)")
bench(lambda: chunked_f(model, concatenated_batch, concatenated_label, compiled=False), name="eager (chunked)")
bench(lambda: opt_f(model, concatenated_batch, concatenated_label), name="compile (non-chunked)")
bench(lambda: chunked_f(model, concatenated_batch, concatenated_label, compiled=True), name="compile (chunked)")
bench(lambda: liger_chunked_f(model, concatenated_batch, concatenated_label, compiled=True), name="compile (chunked modified)")
