
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_jobuser/56/c56qi65nklxraezjwarw7oy6efl6yztw4ixjbfkfnnhhkvcbgckf.py
# Source Nodes: [gather, norm_logits, unnorm_logits_1, unnorm_logits_2], Original ATen: [aten._log_softmax, aten._to_copy, aten.add, aten.gather]
# gather => gather_1
# norm_logits => amax, exp, log, sub, sub_1, sum_1
# unnorm_logits_1 => add
# unnorm_logits_2 => convert_element_type_2
triton_red_fused__log_softmax__to_copy_add_gather_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 131072],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_add_gather_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '620BE4B991CCC88079FD29703288616B3BF60DB5BAD26CDCE55A0789E6044C3A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]

    # Pre-allocate intermediate values
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (128256*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4) # new max
        tmp7 = _tmp5 - tmp6
        tmp8 = tl_math.exp(tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10 = _tmp14 * tmp9
        _tmp14 = tl.where(rmask, tmp10, _tmp14)
        _tmp5 = tl.where(rmask, tmp6, _tmp5)

        tmp11 = tmp4 - _tmp5
        tmp12 = tl_math.exp(tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)

    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, None)

    # Need final normalization using actual max
    _tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    _tmp7 = _tmp5 - _tmp6
    _tmp8 = tl_math.exp(_tmp7)
    _tmp14 = _tmp14 * _tmp8

    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp14, None)
    tmp16 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.full([XBLOCK, 1], 128256, tl.int32)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp16 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp16)
    tl.device_assert((0 <= tmp20) & (tmp20 < 128256), "index out of bounds: 0 <= tmp20 < 128256")
    tmp22 = tl.load(in_ptr0 + (tmp20 + (128256*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp23 = tl.load(in_ptr1 + (tmp20), None, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 - tmp5
    tmp27 = tl_math.log(tmp14)
    tmp28 = tmp26 - tmp27
    tl.store(out_ptr2 + (x0), tmp28, None)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_jobuser/hr/chrs3qaulxfxvkoatkngs4e44ujdpkzw7pedc53qlkkxecgzpgjk.py
# Source Nodes: [_autograd_grad], Original ATen: [aten.nll_loss_backward]
# _autograd_grad => full_default_7, full_default_8, ne_4, scatter, where_2
triton_poi_fused_nll_loss_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '620BE4B991CCC88079FD29703288616B3BF60DB5BAD26CDCE55A0789E6044C3A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131334144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_jobuser/ap/caplcciyb3l3n3p5g73feukddj5b2hsqsaf5tcmmwcz7e7cypfjg.py
# Source Nodes: [ne, sum_2], Original ATen: [aten.ne, aten.sum]
# ne => ne_3
# sum_2 => sum_5
triton_red_fused_ne_sum_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_ne_sum_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '620BE4B991CCC88079FD29703288616B3BF60DB5BAD26CDCE55A0789E6044C3A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + ((1024*((r1 + (8192*x0)) // 1024)) + (r1 % 1024)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tmp2.to(tl.int64)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_jobuser/o6/co6firteuy3vyndz4hpv7vsbdsao43o4u3nkln7x3vkddm2ghlsh.py
# Source Nodes: [ne, sum_2], Original ATen: [aten.ne, aten.sum]
# ne => ne_3
# sum_2 => sum_5
triton_per_fused_ne_sum_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {2: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_ne_sum_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '620BE4B991CCC88079FD29703288616B3BF60DB5BAD26CDCE55A0789E6044C3A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_jobuser/wb/cwbmr3rsuaft4xtnq5u6zdwltlu5kukrv2ymtp5tbza4ieawhfsp.py
# Source Nodes: [_autograd_grad, add__2, chosen_logps, chosen_nll_loss, chosen_nll_loss_1, exp, exp_1, log1p, log1p_1, log_odds, loss, neg, neg_1, or_loss, or_loss_1, ratio, rejected_logps, sigmoid, sub, sub_1, sum_1], Original ATen: [aten.add, aten.div, aten.exp, aten.log, aten.log1p, aten.mean, aten.mul, aten.neg, aten.nll_loss_backward, aten.nll_loss_forward, aten.sigmoid, aten.sub, aten.sum]
# _autograd_grad => full_default_7, full_default_8, ne_4, scatter, where_2
# add__2 => add_10
# chosen_logps => mean
# chosen_nll_loss => full_default_1, ne_1, neg, sum_3, where_1
# chosen_nll_loss_1 => div
# exp => exp_1
# exp_1 => exp_2
# log1p => log1p
# log1p_1 => log1p_1
# log_odds => sub_4
# loss => add_1
# neg => neg_1
# neg_1 => neg_2
# or_loss => mul
# or_loss_1 => div_1
# ratio => log_1
# rejected_logps => mean_1
# sigmoid => sigmoid
# sub => sub_2
# sub_1 => sub_3
# sum_1 => sum_4
triton_red_fused_add_div_exp_log_log1p_mean_mul_neg_nll_loss_backward_nll_loss_forward_sigmoid_sub_sum_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {12: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13), equal_to_1=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_exp_log_log1p_mean_mul_neg_nll_loss_backward_nll_loss_forward_sigmoid_sub_sum_4', 'mutated_arg_names': ['in_ptr6', 'out_ptr2', 'out_ptr3'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 3, 'backend_hash': '620BE4B991CCC88079FD29703288616B3BF60DB5BAD26CDCE55A0789E6044C3A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr0 + (1024 + r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.load(in_ptr5 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp9 = tl.full([1, 1], -100, tl.int64)
        tmp10 = tmp8 != tmp9
        tmp11 = tl.full([1, 1], 0, tl.int64)
        tmp12 = tl.where(tmp10, tmp8, tmp11)
        tmp13 = tl.full([XBLOCK, RBLOCK], 128256, tl.int32)
        tmp14 = tmp12 + tmp13
        tmp15 = tmp12 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp12)
        tl.device_assert(((0 <= tmp16) & (tmp16 < 128256)) | ~(rmask), "index out of bounds: 0 <= tmp16 < 128256")
        tmp18 = -1.0
        tmp19 = tl.load(in_ptr2 + (tmp16 + (128256*r0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr3 + (tmp16), rmask, eviction_policy='evict_last').to(tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp22 - tmp23
        tmp26 = tl_math.log(tmp25)
        tmp27 = tmp24 - tmp26
        tmp28 = -tmp27
        tmp29 = 0.0
        tmp30 = tl.where(tmp10, tmp28, tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask, tmp33, _tmp32)
        tl.store(out_ptr2 + (tl.broadcast_to(tmp16 + (128256*r0), [XBLOCK, RBLOCK])), tmp18, rmask)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp2, None)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp6, None)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tmp34 = tl.load(in_ptr6 + (0))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, 1])
    tmp36 = ks0
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp32 / tmp37
    tmp39 = 1024.0
    tmp40 = tmp2 / tmp39
    tmp41 = tmp6 / tmp39
    tmp42 = tmp40 - tmp41
    tmp43 = tl_math.exp(tmp40)
    tmp44 = -tmp43
    tmp45 = libdevice.log1p(tmp44)
    tmp46 = tl_math.exp(tmp41)
    tmp47 = -tmp46
    tmp48 = libdevice.log1p(tmp47)
    tmp49 = tmp45 - tmp48
    tmp50 = tmp42 - tmp49
    tmp51 = tl.sigmoid(tmp50)
    tmp52 = tl_math.log(tmp51)
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = 0.03125
    tmp56 = tmp54 * tmp55
    tmp57 = tmp38 + tmp56
    tmp58 = tmp35 + tmp57
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp58, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_jobuser/jr/cjrejqhr5cpq6e5lidcwwgqczatcau6qvbp6m5q74xcj6lu2p23z.py
# Source Nodes: [_autograd_grad], Original ATen: [aten.new_zeros, aten.scatter_add]
# _autograd_grad => full_default_6, scatter_add
triton_poi_fused_new_zeros_scatter_add_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[268435456], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_scatter_add_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '620BE4B991CCC88079FD29703288616B3BF60DB5BAD26CDCE55A0789E6044C3A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262668288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_jobuser/jv/cjvvfkldxw3dq4yktqnb7sikiwom3cskefboch3oiytk7hfnzmzg.py
# Source Nodes: [_autograd_grad], Original ATen: [aten.div, aten.new_zeros, aten.scatter_add, aten.slice_backward]
# _autograd_grad => div_7, div_8, full_default_4, full_default_5, full_default_6, scatter_add
triton_poi_fused_div_new_zeros_scatter_add_slice_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_new_zeros_scatter_add_slice_backward_6', 'mutated_arg_names': ['out_ptr2'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '620BE4B991CCC88079FD29703288616B3BF60DB5BAD26CDCE55A0789E6044C3A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp49 = tl.load(in_ptr2 + (x2), None)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp5 = 1024.0
    tmp6 = tmp4 / tmp5
    tmp9 = tmp8 / tmp5
    tmp10 = tmp6 - tmp9
    tmp11 = tl_math.exp(tmp6)
    tmp12 = -tmp11
    tmp13 = libdevice.log1p(tmp12)
    tmp14 = tl_math.exp(tmp9)
    tmp15 = -tmp14
    tmp16 = libdevice.log1p(tmp15)
    tmp17 = tmp13 - tmp16
    tmp18 = tmp10 - tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = 0.03125
    tmp21 = tmp20 / tmp19
    tmp22 = 1.0
    tmp23 = tmp22 - tmp19
    tmp24 = tmp19 * tmp23
    tmp25 = tmp21 * tmp24
    tmp26 = -tmp25
    tmp27 = -tmp26
    tmp28 = tmp15 + tmp22
    tmp29 = tmp27 / tmp28
    tmp30 = -tmp29
    tmp31 = tmp30 * tmp14
    tmp32 = tmp31 + tmp26
    tmp33 = 0.0009765625
    tmp34 = tmp32 * tmp33
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp2, tmp34, tmp35)
    tmp37 = 0.0
    tmp38 = tl.where(tmp2, tmp36, tmp37)
    tmp39 = tmp0 < tmp1
    tmp40 = tmp12 + tmp22
    tmp41 = tmp26 / tmp40
    tmp42 = -tmp41
    tmp43 = tmp42 * tmp11
    tmp44 = tmp43 + tmp25
    tmp45 = tmp44 * tmp33
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp39, tmp45, tmp46)
    tmp48 = tl.where(tmp39, tmp47, tmp37)
    tmp50 = tl.full([XBLOCK], 128256, tl.int32)
    tmp51 = tmp49 + tmp50
    tmp52 = tmp49 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp49)
    tl.device_assert((0 <= tmp53) & (tmp53 < 128256), "index out of bounds: 0 <= tmp53 < 128256")
    tmp55 = tmp38 + tmp48
    tl.atomic_add(out_ptr2 + (tmp53 + (128256*x2)), tmp55, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_jobuser/hu/chuqxdiwbmgsnhgpnoi2hvsaflehjkkuc4s5a3b6q4ql5fz6n6m6.py
# Source Nodes: [_autograd_grad, norm_logits, unnorm_logits_1, unnorm_logits_2], Original ATen: [aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.add, aten.slice_backward, aten.view]
# _autograd_grad => add_7, convert_element_type_4, exp_3, full_default_10, mul_7, sub_6, sum_6, view_6
# norm_logits => log, sub, sub_1
# unnorm_logits_1 => add
# unnorm_logits_2 => convert_element_type_2
triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_add_slice_backward_view_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[2048, 131072],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_add_slice_backward_view_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 1, 'backend_hash': '620BE4B991CCC88079FD29703288616B3BF60DB5BAD26CDCE55A0789E6044C3A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = (xindex // 1024)
    x0 = xindex % 1024
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128256*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = x1
        tmp2 = tl.full([1, 1], 1, tl.int64)
        tmp3 = tmp1 < tmp2
        tmp4 = tl.load(in_ptr1 + (r2 + (128256*x0)), rmask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full([1, 1], -100, tl.int64)
        tmp7 = tmp5 != tmp6
        tmp8 = 1.0
        tmp9 = tl.broadcast_to(ks0, [XBLOCK, RBLOCK])
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp8 / tmp10
        tmp12 = 0.0
        tmp13 = tl.where(tmp7, tmp11, tmp12)
        tmp14 = tmp4 * tmp13
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp3, tmp14, tmp15)
        tmp17 = tl.where(tmp3, tmp16, tmp12)
        tmp18 = tmp0 + tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask, tmp21, _tmp20)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tmp45 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp22 = tl.load(in_ptr0 + (r2 + (128256*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp41 = tl.load(in_ptr3 + (r2 + (128256*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp42 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = x1
        tmp24 = tl.full([1, 1], 1, tl.int64)
        tmp25 = tmp23 < tmp24
        tmp26 = tl.load(in_ptr1 + (r2 + (128256*x0)), rmask & tmp25, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp25, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.full([1, 1], -100, tl.int64)
        tmp29 = tmp27 != tmp28
        tmp30 = 1.0
        tmp31 = tl.broadcast_to(ks0, [XBLOCK, RBLOCK])
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp30 / tmp32
        tmp34 = 0.0
        tmp35 = tl.where(tmp29, tmp33, tmp34)
        tmp36 = tmp26 * tmp35
        tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
        tmp38 = tl.where(tmp25, tmp36, tmp37)
        tmp39 = tl.where(tmp25, tmp38, tmp34)
        tmp40 = tmp22 + tmp39
        tmp43 = tmp41 + tmp42
        tmp44 = tmp43.to(tl.float32)
        tmp46 = tmp44 - tmp45
        tmp48 = tl_math.log(tmp47)
        tmp49 = tmp46 - tmp48
        tmp50 = tl_math.exp(tmp49)
        tmp51 = tmp50 * tmp20
        tmp52 = tmp40 - tmp51
        tmp53 = tmp52.to(tl.float32)
        tl.store(out_ptr1 + (r2 + (128256*x3)), tmp52, rmask)
        tl.store(out_ptr2 + (r2 + (128256*x3)), tmp53, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_jobuser/mj/cmjwmqllj4jgiym7xu4fjcygie4bntkorrsyzko42rrpvulllsvp.py
# Source Nodes: [_autograd_grad, add_], Original ATen: [aten._to_copy, aten.add, aten.sum]
# _autograd_grad => convert_element_type_4, sum_7
# add_ => add_8
triton_red_fused__to_copy_add_sum_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[131072, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_sum_8', 'mutated_arg_names': ['in_ptr1', 'out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '620BE4B991CCC88079FD29703288616B3BF60DB5BAD26CDCE55A0789E6044C3A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128256
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_jobuser/ai/caicvkjxws4w5xvqylrri46njvqb3xiek57eg7d7ao57rpa6fqjn.py
# Source Nodes: [add__1], Original ATen: [aten.add]
# add__1 => add_9
triton_poi_fused_add_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_9', 'mutated_arg_names': ['in_out_ptr0', 'in_ptr0', 'out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '620BE4B991CCC88079FD29703288616B3BF60DB5BAD26CDCE55A0789E6044C3A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98500608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
    args.clear()
    # assert_size_stride(arg0_1, (128256, ), (1, ))
    # assert_size_stride(arg1_1, (2, 1024, 768), (786432, 768, 1))
    # assert_size_stride(arg2_1, (128256, 768), (768, 1))
    # assert_size_stride(arg3_1, (2, 1024), (1024, 1))
    # assert_size_stride(arg4_1, (32, 1024), (1024, 1))
    # assert_size_stride(arg5_1, (128256, ), (1, ))
    # assert_size_stride(arg6_1, (128256, 768), (768, 1))
    # assert_size_stride(arg7_1, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 128256), (128256, 1), torch.bfloat16)
        # Source Nodes: [unnorm_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg1_1, (2048, 768), (768, 1), 0), reinterpret_tensor(arg2_1, (768, 128256), (1, 768), 0), out=buf0)
        buf1 = empty_strided_cuda((2, 1024, 1), (1024, 1, 2048), torch.float32)
        buf2 = empty_strided_cuda((2, 1024, 1), (1024, 1, 2048), torch.float32)
        buf3 = empty_strided_cuda((2, 1024, 1), (1024, 1, 2048), torch.float32)
        # Source Nodes: [gather, norm_logits, unnorm_logits_1, unnorm_logits_2], Original ATen: [aten._log_softmax, aten._to_copy, aten.add, aten.gather]
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax__to_copy_add_gather_0.run(buf0, arg0_1, arg3_1, buf1, buf2, buf3, 2048, 128256, grid=grid(2048), stream=stream0)
        buf10 = empty_strided_cuda((1024, 128256), (128256, 1), torch.float32)
        # Source Nodes: [_autograd_grad], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_1.run(buf10, 131334144, grid=grid(131334144), stream=stream0)
        buf12 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Source Nodes: [ne, sum_2], Original ATen: [aten.ne, aten.sum]
        triton_red_fused_ne_sum_2.run(arg4_1, buf12, 2, 8192, grid=grid(2), stream=stream0)
        del arg4_1
        buf13 = empty_strided_cuda((), (), torch.int64)
        # Source Nodes: [ne, sum_2], Original ATen: [aten.ne, aten.sum]
        triton_per_fused_ne_sum_3.run(buf12, buf13, 1, 2, grid=grid(1), stream=stream0)
        del buf12
    u0 = buf13.item()
    buf14 = None
    del buf13
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf4 = empty_strided_cuda((1, ), (1, ), torch.float32)
        buf5 = empty_strided_cuda((1, ), (1, ), torch.float32)
        # Source Nodes: [_autograd_grad, add__2, chosen_logps, chosen_nll_loss, chosen_nll_loss_1, exp, exp_1, log1p, log1p_1, log_odds, loss, neg, neg_1, or_loss, or_loss_1, ratio, rejected_logps, sigmoid, sub, sub_1, sum_1], Original ATen: [aten.add, aten.div, aten.exp, aten.log, aten.log1p, aten.mean, aten.mul, aten.neg, aten.nll_loss_backward, aten.nll_loss_forward, aten.sigmoid, aten.sub, aten.sum]
        triton_red_fused_add_div_exp_log_log1p_mean_mul_neg_nll_loss_backward_nll_loss_forward_sigmoid_sub_sum_4.run(buf3, arg3_1, buf0, arg0_1, buf1, buf2, arg7_1, buf4, buf5, buf10, arg7_1, u0, 1, 1024, grid=grid(1), stream=stream0)
        del arg7_1
        del buf3
        buf8 = empty_strided_cuda((2, 1024, 128256), (131334144, 128256, 1), torch.float32)
        # Source Nodes: [_autograd_grad], Original ATen: [aten.new_zeros, aten.scatter_add]
        triton_poi_fused_new_zeros_scatter_add_5.run(buf8, 262668288, grid=grid(262668288), stream=stream0)
        # Source Nodes: [_autograd_grad], Original ATen: [aten.div, aten.new_zeros, aten.scatter_add, aten.slice_backward]
        triton_poi_fused_div_new_zeros_scatter_add_slice_backward_6.run(buf4, buf5, arg3_1, buf8, 2048, grid=grid(2048), stream=stream0)
        del buf4
        del buf5
        buf16 = empty_strided_cuda((2, 1024, 128256), (131334144, 128256, 1), torch.float32)
        buf17 = empty_strided_cuda((2048, 128256), (128256, 1), torch.bfloat16)
        # Source Nodes: [_autograd_grad, norm_logits, unnorm_logits_1, unnorm_logits_2], Original ATen: [aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.add, aten.slice_backward, aten.view]
        triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_add_slice_backward_view_7.run(buf8, buf10, arg3_1, buf0, arg0_1, buf1, buf2, buf16, buf17, u0, 2048, 128256, grid=grid(2048), stream=stream0)
        del arg0_1
        del arg3_1
        del buf0
        del buf1
        del buf10
        del buf2
        del buf8
        buf18 = empty_strided_cuda((2048, 768), (768, 1), torch.bfloat16)
        # Source Nodes: [_autograd_grad], Original ATen: [aten.mm]
        extern_kernels.mm(buf17, reinterpret_tensor(arg2_1, (128256, 768), (768, 1), 0), out=buf18)
        del arg2_1
        # Source Nodes: [_autograd_grad, add_], Original ATen: [aten._to_copy, aten.add, aten.sum]
        triton_red_fused__to_copy_add_sum_8.run(buf16, arg5_1, arg5_1, 128256, 2048, grid=grid(128256), stream=stream0)
        del arg5_1
        del buf16
        buf20 = empty_strided_cuda((128256, 768), (768, 1), torch.bfloat16)
        # Source Nodes: [_autograd_grad], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (128256, 2048), (1, 128256), 0), reinterpret_tensor(arg1_1, (2048, 768), (768, 1), 0), out=buf20)
        del arg1_1
        del buf17
        buf24 = buf20; del buf20  # reuse
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_9.run(buf24, arg6_1, arg6_1, 98500608, grid=grid(98500608), stream=stream0)
        del arg6_1
        del buf24
    return (reinterpret_tensor(buf18, (2, 1024, 768), (786432, 768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((2, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((128256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((2, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg4_1 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg5_1 = rand_strided((128256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((128256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
