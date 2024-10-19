import torch
import triton
import triton.language as tl
from typing import List
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_device("cuda")

B, T, D, V = 32, 1024, 256, 4096
model = nn.Linear(D, V).to(torch.bfloat16)
nll = nn.NLLLoss(reduction="sum")
ce = nn.CrossEntropyLoss()
chosen = torch.randn(B, T, D, requires_grad=True, dtype=torch.bfloat16)
rejected = torch.randn(B, T, D, requires_grad=True, dtype=torch.bfloat16)
chosen_label = torch.randint(0, V, (B, T)).to(torch.int64)
rejected_label = torch.randint(0, V, (B, T)).to(torch.int64)


def f(m, x, label):
    out = ce(m(x).view(-1, V), label.view(-1))
    return out


def f2(m, x, label):
    logits = m(x)
    out = nll(F.log_softmax(logits, dim=-1).view(-1, V), label.view(-1))
    return out


def odds_ratio_loss(chosen_logps, rejected_logps, beta=1.0):
    log_odds = (chosen_logps - rejected_logps) - (
        torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
    )
    sig_ratio = F.sigmoid(log_odds)
    ratio = torch.log(sig_ratio)
    losses = beta * ratio
    return losses.sum()


def f3(m, chosen, rejected, chosen_label, rejected_label):
    chosen_norm_logits = F.log_softmax(m(chosen), dim=-1)
    rejected_norm_logits = F.log_softmax(m(rejected), dim=-1)

    chosen_nll_loss = nll(chosen_norm_logits.view(-1, V), chosen_label.view(-1))

    chosen_logps = chosen_norm_logits.gather(-1, chosen_label.unsqueeze(2)).squeeze(2)
    rejected_logps = rejected_norm_logits.gather(
        -1, rejected_label.unsqueeze(2)
    ).squeeze(2)

    or_loss = odds_ratio_loss(chosen_logps, rejected_logps)

    print(chosen_nll_loss, or_loss)
    loss = chosen_nll_loss + or_loss
    loss.backward()
    return loss


def f4(m, batch, label):
    len_chosen = batch.shape[0] // 2
    unnorm_logits = m(batch)
    unnorm_logits = unnorm_logits.float()
    concatenated_logits = F.log_softmax(unnorm_logits, dim=-1)
    chosen_nll_loss = nll(
        concatenated_logits[:len_chosen].view(-1, V), label[:len_chosen].view(-1)
    )

    all_logps = concatenated_logits.gather(-1, label.unsqueeze(2)).squeeze(2)
    chosen_logps = all_logps[:len_chosen]
    rejected_logps = all_logps[len_chosen:]

    or_loss = odds_ratio_loss(chosen_logps, rejected_logps)
    loss = chosen_nll_loss + or_loss
    loss.backward()
    return loss


class ChunkedORPO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, target, bias=None, compiled=True):
        CHUNK_SIZE = 1024

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
            (chunk_grad_input, chunk_grad_weight, chunk_grad_bias), chunk_loss = (
                torch.func.grad_and_value(compute_orpo_loss, argnums=(0, 1, 2))(
                    input_chunk, weight, bias, target_chunk
                )
            )
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
            grad_chosen_inputs.append(grad_input[:chosen_target_chunk.shape[0]])
            grad_rejected_inputs.append(grad_input[chosen_target_chunk.shape[0]:])

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


def f5(m, batch, label, compiled=True):
    out = ChunkedORPO.apply(batch.view(-1, D), m.weight, label.view(-1), m.bias, compiled)
    out.backward()
    return out

# print(f(model, chosen, label))
# print(f2(model, chosen, label))
# print("f3", f3(model, chosen, rejected, chosen_label, rejected_label))


concatenated_batch = torch.cat([chosen, rejected], dim=0)
concatenated_label = torch.cat([chosen_label, rejected_label], dim=0)
print("f4", f4(model, concatenated_batch, concatenated_label))

print("f5", f5(model, concatenated_batch, concatenated_label))
