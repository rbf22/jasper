"""Mamba-3 MIMO Layer (tt-nn native).

Implements the three core Mamba-3 improvements:
1. Exponential-trapezoidal discretization (more expressive recurrence)
2. Complex-valued SSM via RoPE (richer state tracking)
3. MIMO (multi-input multi-output) for better performance

Forward uses quadratic (T×T) attention with R² cross-rank pairs.
Backward is manual, computed on device.
"""

import math
import os
import time
from typing import Dict, Tuple

import torch
import ttnn

# Per-section timing (enabled by setting M3_PROFILE=1 in env)
_M3_PROFILE = os.environ.get("M3_PROFILE", "0") == "1"
_M3_TIMINGS: Dict[str, float] = {}

def _t(name, t0=None):
    """Lightweight section timer. Returns current time for start, records elapsed for end."""
    if not _M3_PROFILE:
        return None
    now = time.perf_counter()
    if t0 is not None:
        dt = now - t0
        _M3_TIMINGS[name] = _M3_TIMINGS.get(name, 0.0) + dt
    return now

from model_ttnn import ModelConfig, to_device, TTRMSNorm

# Custom kernel support
from ttnn._ttnn.program_descriptor import VectorUInt32

_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")

# fp32 accumulation in the matmul destination register while keeping tensors
# stored as bf16 (no extra DRAM traffic vs. materializing fp32 tensors).
# Blackhole does not have the Wormhole fp32_dest_acc_en rounding erratum
# (see ttnn.matmul docstring), so this is safe here.
#
# Applied only to the QK / grad_QK matmuls that directly feed the decay-matrix
# backward (grad_L_accum = grad_QK * QK_raw -> row_sum - col_sum -> grad_A_log),
# which is the one place in this layer with a catastrophic-cancellation
# reduction. Measured effect (tile-aligned, grad_A_log rel_err vs. fp32
# reference): T=32 0.069->0.067 (noise), T=64 0.142->0.085 (real reduction).
# Broadening this to the other four matmuls in the R^2 loop (out_rq
# accumulation, grad_V_rk, grad_Q_rq, grad_K_rk) was measured and gave no
# further improvement anywhere, because fp32_dest_acc_en only raises the
# precision of the accumulator, not of the bf16-quantized Q/K/V tiles being
# read — so it doesn't help ops that aren't themselves cancellation-sensitive.
# Not applied elsewhere to avoid paying HiFi4's extra compute cost for zero
# accuracy return.
_HIFI_FP32_ACC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)


class TTMamba3Layer:
    """Mamba-3 MIMO layer using tt-nn operations."""

    @staticmethod
    def _fused_diag_corr_forward(Q_thrn, K_thrn, V_proj_thrp, gamma, B, T, H, R, N, P, device):
        """Fused diagonal correction via custom kernel: out = gamma * (Q @ K^T) @ V

        Replaces 5+ ttnn ops (transpose, matmul, mul, matmul, permute) with a single
        generic_op call. 32-45x faster than the ttnn equivalent.

        Args:
            Q_thrn: (B,T,H,R,N) bf16 TILE on device
            K_thrn: (B,T,H,R,N) bf16 TILE on device
            V_proj_thrp: (B,T,H,R,P) bf16 TILE on device
            gamma: (B,T,H) bf16 TILE on device
        Returns:
            qkv_diag: (B,T,H,R,P) bf16 TILE on device
        """
        gamma_4d = ttnn.reshape(gamma, [B * T * H, 1, 1, 1])
        Nt = (N + 31) // 32
        Pt = (P + 31) // 32
        Rt = (R + 31) // 32
        batch_size = B * T * H
        tiles_per_batch = Rt * Pt
        tiles_per_batch_qk = Rt * Nt

        out = ttnn.empty([B, T, H, R, P], dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=device)

        grid = device.compute_with_storage_grid_size()
        num_cores_x, num_cores_y = grid.x, grid.y
        num_cores_total = num_cores_x * num_cores_y
        num_cores = min(batch_size, num_cores_total)
        bpc_base = batch_size // num_cores
        bpc_rem = batch_size % num_cores

        all_cores = ttnn.CoreRangeSet([
            ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                          ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))
        ])

        tile_bytes = 32 * 32 * 2  # bf16
        cb_q, cb_k, cb_v, cb_gamma = 0, 1, 2, 3
        cb_qk, cb_qk_g, cb_out = 4, 5, 16

        def _cb(idx, n):
            return ttnn.CBDescriptor(
                total_size=n * tile_bytes, core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)])

        cbs = [_cb(cb_q, 2), _cb(cb_k, 2), _cb(cb_v, 2), _cb(cb_gamma, 1),
               _cb(cb_qk, 1), _cb(cb_qk_g, 1), _cb(cb_out, 2)]

        # Compile-time args with TensorAccessorArgs
        reader_ct = []
        for t in [Q_thrn, K_thrn, V_proj_thrp, gamma_4d]:
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        writer_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())

        # Runtime args per core
        reader_rt, writer_rt, compute_rt = [], [], []
        bs = 0
        for i in range(num_cores_total):
            cx, cy = i // num_cores_y, i % num_cores_y
            coord = ttnn.CoreCoord(cx, cy)
            ntpc = bpc_base + (1 if i < bpc_rem else 0) if i < num_cores else 0
            reader_rt.append((coord, VectorUInt32([
                Q_thrn.buffer_address(), K_thrn.buffer_address(),
                V_proj_thrp.buffer_address(), gamma_4d.buffer_address(),
                ntpc, bs, tiles_per_batch_qk, 1])))
            writer_rt.append((coord, VectorUInt32([
                out.buffer_address(), ntpc, bs, tiles_per_batch])))
            compute_rt.append((coord, VectorUInt32([ntpc])))
            bs += ntpc

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/diag_corr_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(reader_ct),
            runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/diag_corr_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(writer_ct),
            runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor())
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/diag_corr_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32([Nt, Pt]),
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True))

        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute],
                                         semaphores=[], cbs=cbs)
        return ttnn.generic_op(io_tensors=[Q_thrn, K_thrn, V_proj_thrp, gamma_4d, out],
                               program_descriptor=program)

    @staticmethod
    def _fused_rope_forward(t0, t1, cos, sin, B, T, R, H, n_angles, device):
        """Fused RoPE rotation via custom kernel: rot_0 = t0*cos - t1*sin, rot_1 = t0*sin + t1*cos

        Replaces 6 ttnn ops (4 muls + 1 sub + 1 add) with a single generic_op call.

        Args:
            t0: (B,T,R,H,n_angles) bf16 TILE — first half of deinterleaved tensor
            t1: (B,T,R,H,n_angles) bf16 TILE — second half
            cos: (B,T,H,n_angles) bf16 TILE — cos (broadcast over R by kernel)
            sin: (B,T,H,n_angles) bf16 TILE — sin (broadcast over R by kernel)
        Returns:
            rot_0: (B,T,R,H,n_angles) bf16 TILE
            rot_1: (B,T,R,H,n_angles) bf16 TILE
        """
        # Number of tiles: t0 has B*T*R tiles (H,n_angles packed in 32x32)
        # cos/sin have B*T tiles (broadcast over R in reader kernel)
        batch_size = B * T * R

        rot_0 = ttnn.empty([B, T, R, H, n_angles], dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device)
        rot_1 = ttnn.empty([B, T, R, H, n_angles], dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device)

        grid = device.compute_with_storage_grid_size()
        num_cores_x, num_cores_y = grid.x, grid.y
        num_cores_total = num_cores_x * num_cores_y
        num_cores = min(batch_size, num_cores_total)
        bpc_base = batch_size // num_cores
        bpc_rem = batch_size % num_cores

        all_cores = ttnn.CoreRangeSet([
            ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                          ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))
        ])

        tile_bytes = 32 * 32 * 2  # bf16
        cb_t0, cb_t1, cb_cos, cb_sin = 0, 1, 2, 3
        cb_p0, cb_p1, cb_p2, cb_p3 = 4, 5, 6, 7
        cb_rot0, cb_rot1 = 16, 17

        def _cb(idx, n):
            return ttnn.CBDescriptor(
                total_size=n * tile_bytes, core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(
                    buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)])

        # Input CBs: size 2 (each tile pushed twice for the two products)
        # Intermediate CBs: size 1
        # Output CBs: size 2
        cbs = [_cb(cb_t0, 2), _cb(cb_t1, 2), _cb(cb_cos, 2), _cb(cb_sin, 2),
               _cb(cb_p0, 1), _cb(cb_p1, 1), _cb(cb_p2, 1), _cb(cb_p3, 1),
               _cb(cb_rot0, 2), _cb(cb_rot1, 2)]

        # Compile-time args with TensorAccessorArgs
        reader_ct = []
        for t in [t0, t1, cos, sin]:
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        writer_ct = list(ttnn.TensorAccessorArgs(rot_0).get_compile_time_args())
        writer_ct.extend(ttnn.TensorAccessorArgs(rot_1).get_compile_time_args())

        # Runtime args per core
        reader_rt, writer_rt, compute_rt = [], [], []
        bs = 0
        for i in range(num_cores_total):
            cx, cy = i // num_cores_y, i % num_cores_y
            coord = ttnn.CoreCoord(cx, cy)
            ntpc = bpc_base + (1 if i < bpc_rem else 0) if i < num_cores else 0
            reader_rt.append((coord, VectorUInt32([
                t0.buffer_address(), t1.buffer_address(),
                cos.buffer_address(), sin.buffer_address(),
                ntpc, bs, R])))
            writer_rt.append((coord, VectorUInt32([
                rot_0.buffer_address(), rot_1.buffer_address(),
                ntpc, bs, 1])))  # 1 tile per batch element
            compute_rt.append((coord, VectorUInt32([ntpc])))
            bs += ntpc

        reader = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/rope_fwd_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(reader_ct),
            runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor())
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/rope_fwd_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32(writer_ct),
            runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor())
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/rope_fwd_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores, compile_time_args=VectorUInt32([]),
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False))

        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute],
                                         semaphores=[], cbs=cbs)
        ttnn.generic_op(io_tensors=[t0, t1, cos, sin, rot_0, rot_1],
                        program_descriptor=program)
        return rot_0, rot_1

    @staticmethod
    def _flip(tensor, dims):
        """Flip a ttnn tensor along given dims (host-side, for backward pass only)."""
        host = ttnn.to_torch(tensor)
        host_flipped = torch.flip(host, dims=dims)
        result = ttnn.from_torch(
            host_flipped,
            dtype=tensor.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=tensor.device(),
        )
        return result

    @staticmethod
    def _reverse_cumsum(tensor, dim):
        """Reverse cumsum entirely on device: reverse_cumsum(x) = sum(x) - cumsum(x) + x.

        Replaces the _flip + cumsum + _flip pattern that required two host-device
        round trips.  The identity holds for any dim and any dtype.
        """
        device = tensor.device()
        dtype = tensor.dtype
        # Total sum along dim, broadcast back to the original shape
        total = ttnn.sum(tensor, dim=dim, keepdim=True)
        # Expand the reduced dim back to the original size
        expand_shape = list(int(tensor.shape[i]) for i in range(len(tensor.shape)))
        total = ttnn.expand(total, expand_shape)
        forward_cs = ttnn.cumsum(tensor, dim=dim)
        # reverse_cumsum = total - cumsum + tensor
        return ttnn.add(ttnn.sub(total, forward_cs), tensor)

    def __init__(self, config: ModelConfig, device):
        self.config = config
        self.device = device
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.headdim = config.headdim
        self.nheads = config.nheads_m3
        self.d_state = config.d_state_m3
        self.R = config.mimo_rank
        self.ngroups = config.ngroups
        self.num_rope_angles = config.num_rope_angles
        # ttnn.expand can only broadcast size-1 dims, so the GQA expansion in
        # forward() only works for ngroups == 1. The backward reduction assumes
        # a general grouping the forward cannot produce.
        assert self.ngroups == 1, (
            f"ngroups={self.ngroups}: real GQA repeat is not implemented; "
            "only ngroups == 1 is supported.")
        self.split_tensor_size = int(self.d_state * config.rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1

        # in_proj: (d_model -> d_in_proj), order: [z, x, B, C, dt, A, trap, angles]
        d_in_proj = (2 * self.d_inner
                     + 2 * self.d_state * self.ngroups * self.R
                     + 3 * self.nheads
                     + self.num_rope_angles)
        w = torch.randn(self.d_model, d_in_proj, dtype=torch.bfloat16) * 0.02
        self.in_proj_weight = to_device(w, device)

        # dt_bias: (nheads,) fp32
        dt_min, dt_max = 0.001, 0.1
        _dt = torch.exp(torch.rand(self.nheads, dtype=torch.float32)
                        * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        _dt = torch.clamp(_dt, min=1e-4)
        dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = to_device(dt_bias, device, dtype=ttnn.float32)

        # A_log: (nheads,) fp32 — A = -softplus(A_log)
        self.A_log = to_device(torch.randn(self.nheads, dtype=torch.float32) * 0.1,
                               device, dtype=ttnn.float32)

        # D skip: (nheads,) fp32, init 1
        self.D = to_device(torch.ones(self.nheads, dtype=torch.float32),
                           device, dtype=ttnn.float32)

        # B/C biases: (nheads, R, d_state) fp32, init 0
        self.B_bias = to_device(torch.zeros(self.nheads, self.R, self.d_state, dtype=torch.float32),
                                device, dtype=ttnn.float32)
        self.C_bias = to_device(torch.zeros(self.nheads, self.R, self.d_state, dtype=torch.float32),
                                device, dtype=ttnn.float32)

        # QKNorm (RMSNorm on B and C along d_state)
        self.B_norm = TTRMSNorm(self.d_state, device)
        self.C_norm = TTRMSNorm(self.d_state, device)

        # MIMO projections: (nheads, R, headdim) fp32.
        # Every other tensor these multiply (V, z) is broadcast identically
        # across R, so if MIMO_V/Z/O ever have identical values across the R
        # axis, all R "parallel SSMs" compute the exact same thing and receive
        # the exact same gradient — that symmetry is a fixed point of gradient
        # descent and is never broken. Random per-rank init is required to let
        # the R ranks specialize at all; a constant init here is a standing
        # capacity bug, not just a validation-hiding one (it makes the MIMO
        # mechanism permanently degenerate to a single effective rank).
        self.MIMO_V = to_device(
            (torch.ones(self.nheads, self.R, self.headdim, dtype=torch.float32)
             + torch.randn(self.nheads, self.R, self.headdim, dtype=torch.float32) * 0.02) / self.R,
            device, dtype=ttnn.float32)
        self.MIMO_Z = to_device(
            torch.ones(self.nheads, self.R, self.headdim, dtype=torch.float32)
            + torch.randn(self.nheads, self.R, self.headdim, dtype=torch.float32) * 0.02,
            device, dtype=ttnn.float32)
        self.MIMO_O = to_device(
            (torch.ones(self.nheads, self.R, self.headdim, dtype=torch.float32)
             + torch.randn(self.nheads, self.R, self.headdim, dtype=torch.float32) * 0.02) / self.R,
            device, dtype=ttnn.float32)

        # out_proj: (d_inner -> d_model)
        self.out_proj_weight = to_device(
            torch.randn(self.d_inner, self.d_model, dtype=torch.bfloat16) * 0.02, device)

        # Pre-compute small constant tensors to avoid from_torch on every
        # forward/backward call (each from_torch is a host-device transfer).
        self._ones_1 = ttnn.from_torch(
            torch.ones(1, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self._pi_tt = ttnn.from_torch(
            torch.tensor([math.pi], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self._inv_pi = ttnn.from_torch(
            torch.tensor([1.0 / math.pi], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Pre-compute parameter base tensors (permute + reshape + typecast),
        # ready for broadcasting mul/add without expand.
        # MIMO_V/Z: (H,R,P) -> permute -> (R,H,P) -> reshape -> (1,1,R,H,P) bf16
        self._MV_base = ttnn.typecast(
            ttnn.reshape(ttnn.permute(self.MIMO_V, [1, 0, 2]), [1, 1, self.R, self.nheads, self.headdim]),
            ttnn.bfloat16)
        self._MZ_base = ttnn.typecast(
            ttnn.reshape(ttnn.permute(self.MIMO_Z, [1, 0, 2]), [1, 1, self.R, self.nheads, self.headdim]),
            ttnn.bfloat16)
        # MIMO_O: (H,R,P) -> reshape -> (1,H,1,R,P) bf16  [for (B,H,T,R,P) layout]
        self._MO_base = ttnn.typecast(
            ttnn.reshape(self.MIMO_O, [1, self.nheads, 1, self.R, self.headdim]),
            ttnn.bfloat16)
        # MIMO_O for (B,H,R,T,P) layout: (H,R,P) -> reshape -> (1,H,R,1,P) bf16
        self._MO_base_rtp = ttnn.typecast(
            ttnn.reshape(self.MIMO_O, [1, self.nheads, self.R, 1, self.headdim]),
            ttnn.bfloat16)
        # B/C bias: (H,R,N) -> permute -> (R,H,N) -> reshape -> (1,1,R,H,N) bf16
        self._B_bias_base = ttnn.typecast(
            ttnn.reshape(ttnn.permute(self.B_bias, [1, 0, 2]), [1, 1, self.R, self.nheads, self.d_state]),
            ttnn.bfloat16)
        self._C_bias_base = ttnn.typecast(
            ttnn.reshape(ttnn.permute(self.C_bias, [1, 0, 2]), [1, 1, self.R, self.nheads, self.d_state]),
            ttnn.bfloat16)
        # D: (H,) -> reshape -> (1,H,1,1,1) bf16
        self._D_base = ttnn.typecast(
            ttnn.reshape(self.D, [1, self.nheads, 1, 1, 1]),
            ttnn.bfloat16)

        self._cache = {}

    # -- Helpers --

    def _get_strict_mask_exp(self, T, R, device):
        """(1,1,T*R,T*R) strict causal mask expanded by R. Cached per (T,R)."""
        TR = T * R
        cache_key = ("strict_mask_exp", T, R)
        if cache_key in self._cache:
            return self._cache[cache_key]
        mask_C = torch.tril(torch.ones(T, T, dtype=torch.float32), diagonal=-1)
        mask_exp = mask_C.unsqueeze(1).unsqueeze(-1).expand(
            T, R, T, R).reshape(TR, TR)
        result = ttnn.from_torch(
            mask_exp.reshape(1, 1, TR, TR),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self._cache[cache_key] = result
        return result

    def _get_strict_mask_4d(self, T, device):
        """(1,1,T,T) strict causal mask. Cached per T."""
        cache_key = ("strict_mask_4d", T)
        if cache_key in self._cache:
            return self._cache[cache_key]
        mask_C = torch.tril(torch.ones(T, T, dtype=torch.float32), diagonal=-1)
        result = ttnn.from_torch(
            mask_C.reshape(1, 1, T, T),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self._cache[cache_key] = result
        return result

    def _shift_by_one(self, tensor, B, T, H, device):
        """Shift (B, T, H) by 1 along T: prepend zeros, remove last."""
        zeros_prefix = ttnn.zeros((B, 1, H), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=device)
        tensor_shifted = ttnn.concat(
            [zeros_prefix, ttnn.slice(tensor, [0, 0, 0], [B, T - 1, H])], dim=1)
        return tensor_shifted

    def _unshift_by_one(self, grad, B, T, H, device):
        """Inverse of _shift_by_one: remove first element, append zeros."""
        zeros_suffix = ttnn.zeros((B, 1, H), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=device)
        return ttnn.concat(
            [ttnn.slice(grad, [0, 1, 0], [B, T, H]), zeros_suffix], dim=1)

    def _apply_rope(self, tensor, cos, sin, B, T, R, H, N):
        """Apply RoPE to tensor (B, T, R, H, N) using cos/sin (B, T, H, n_angles).
        Rotates first split_tensor_size dims as pairs, leaves rest unchanged.

        Output is in DEINTERLEAVED layout: [a0',a1',...,a_{n-1}',b0',...,b_{n-1}',pass...]
        instead of interleaved [a0',b0',a1',b1',...]. This skips 3 ops (reshape,
        permute, reshape) per call. The attention matmul (Q@K^T) is layout-
        invariant as long as Q and K use the same layout, so this is safe.

        Uses a fused custom kernel for the element-wise rotation (6 ops -> 1 kernel).

        Also returns the deinterleaved original (tensor_rot_deint) for caching
        and reuse in the backward pass."""
        split = self.split_tensor_size
        n_angles = self.num_rope_angles
        device = self.device

        # Slice rotated and passthrough parts on last dim
        tensor_rot = ttnn.slice(tensor, [0, 0, 0, 0, 0], [B, T, R, H, split])
        tensor_pass = ttnn.slice(tensor, [0, 0, 0, 0, split], [B, T, R, H, N])

        # Separate pairs via permute: (B,T,R,H,n_angles,2) -> (B,T,R,H,2,n_angles)
        # -> reshape to (B,T,R,H,split) gives [a0,a1,...,a15,b0,b1,...,b15]
        tensor_rot = ttnn.reshape(tensor_rot, [B, T, R, H, n_angles, 2])
        tensor_rot = ttnn.permute(tensor_rot, [0, 1, 2, 3, 5, 4])
        tensor_rot = ttnn.reshape(tensor_rot, [B, T, R, H, split])

        # Contiguous slices: t0 = first half, t1 = second half
        t0 = ttnn.slice(tensor_rot, [0, 0, 0, 0, 0], [B, T, R, H, n_angles])
        t1 = ttnn.slice(tensor_rot, [0, 0, 0, 0, n_angles], [B, T, R, H, split])

        # cos/sin: (B,T,H,n_angles) — kernel handles R broadcast internally
        cos_4d = ttnn.reshape(cos, [B, T, H, n_angles])
        sin_4d = ttnn.reshape(sin, [B, T, H, n_angles])

        # Fused rotation: rot_0 = t0*cos - t1*sin, rot_1 = t0*sin + t1*cos
        rot_0, rot_1 = self._fused_rope_forward(
            t0, t1, cos_4d, sin_4d, B, T, R, H, n_angles, device)

        # Concat WITHOUT interleaving back: [a0',...,a_{n-1}',b0',...,b_{n-1}',pass...]
        # The attention matmul is layout-invariant, so we keep deinterleaved layout.
        out = ttnn.concat([rot_0, rot_1], dim=-1)  # (B,T,R,H,split) — deinterleaved

        tensor_out = ttnn.concat([out, tensor_pass], dim=-1)
        return tensor_out, tensor_rot

    def _causal_mask(self, B, H, T, dtype, torch_dtype):
        """(B, H, T, T) lower-triangular boolean-style mask. Cached per (B,H,T,dtype)."""
        cache_key = ("causal_mask", B, H, T, dtype)
        if cache_key in self._cache:
            return self._cache[cache_key]
        mask = ttnn.from_torch(torch.ones(T, T, dtype=torch_dtype),
                               dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        mask = ttnn.tril(mask, diagonal=0)
        result = ttnn.expand(ttnn.reshape(mask, [1, 1, T, T]), [B, H, T, T])
        self._cache[cache_key] = result
        return result

    def _neg_inf_4d(self, B, H, T):
        """(B, H, T, T) fp32 tensor filled with -1e4. Cached per (B,H,T)."""
        cache_key = ("neg_4d", B, H, T)
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = ttnn.from_torch(torch.full((B, H, T, T), -1e4, dtype=torch.float32),
                                 dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        self._cache[cache_key] = result
        return result

    def _ones_tensor(self, shape, dtype=ttnn.bfloat16):
        """All-ones tensor of given shape/dtype. Cached per (shape, dtype)."""
        cache_key = ("ones", tuple(shape), dtype)
        if cache_key in self._cache:
            return self._cache[cache_key]
        torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
        result = ttnn.from_torch(torch.ones(shape, dtype=torch_dtype),
                                 dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        self._cache[cache_key] = result
        return result

    def _compute_decay_matrix(self, ADT, B, H, T):
        """L[b,h,t,s] = exp(sum(ADT[b,h,s+1:t+1])) for s<=t, 0 for s>t.

        Computed in fp32. The cumsum runs the length of the sequence and its
        result is exponentiated, so bf16 error in the exponent is amplified;
        the matching backward reduction also cancels catastrophically in bf16
        (grad_A_log rel_err reached 0.43 at T=64 before this).

        Returns (L_bf16, L_fp32, ADT_cumsum) — the fp32 copy is kept for the
        backward, and ADT_cumsum is returned so the attention can compute
        exp(cs) / exp(-cs) directly without extracting from L.
        """
        device = self.device
        ADT32 = ttnn.typecast(ADT, ttnn.float32)
        ADT_cumsum = ttnn.cumsum(ADT32, dim=-1)

        ADT_cs_t = ttnn.expand(ttnn.reshape(ADT_cumsum, [B, H, T, 1]), [B, H, T, T])
        ADT_cs_s = ttnn.expand(ttnn.reshape(ADT_cumsum, [B, H, 1, T]), [B, H, T, T])
        segsum = ttnn.sub(ADT_cs_t, ADT_cs_s)

        mask_4d = self._causal_mask(B, H, T, ttnn.float32, torch.float32)
        # Masked positions must be filled with a large negative value *before*
        # the exp. Filling with 0 would give exp(0)=1 in the upper triangle
        # (a non-causal layer), and masking after the exp overflows because
        # segsum > 0 there (ADT < 0 => cumsum is decreasing).
        neg_4d = self._neg_inf_4d(B, H, T)
        segsum_masked = ttnn.where(mask_4d, segsum, neg_4d)
        L32 = ttnn.exp(segsum_masked)
        return ttnn.typecast(L32, ttnn.bfloat16), L32, ADT_cumsum

    def _full_tt_attention(self, Q_perm, K_perm, V_perm, ADT_cumsum, B, H, R, T, N, P, device):
        """Full T×T MIMO attention — 2 matmuls, no sequential loop.

        Folds R into the T dimension and folds the decay matrix L into Q and K
        to avoid the expensive L expansion (which takes 16ms vs 0.1ms for the matmul).

        L[t,s] = exp(cs_t - cs_s) for s < t, 0 for s >= t
        So QK_scaled[t,s] = Q[t] @ K[s]^T * L[t,s]
                         = (exp(cs_t) * Q[t]) @ (exp(-cs_s) * K[s])^T  for s < t

        By pre-scaling Q by exp(cs) and K by exp(-cs), the matmul directly
        produces the decayed QK without needing an element-wise mul with a
        (B,H,T*R,T*R) tensor. The strict causality is enforced by zeroing
        the diagonal and upper triangle after the matmul.

        Args:
            Q_perm: (B,H,R,T,N) bf16 on device (post-RoPE, post-scaling)
            K_perm: (B,H,R,T,N) bf16 on device (post-RoPE, post-scaling)
            V_perm: (B,H,R,T,P) bf16 on device
            ADT_cumsum: (B,H,T) fp32 cumsum of ADT — cs_t for each timestep
        Returns:
            out_attn: (B,H,R,T,P) strictly causal attention output
            cache: dict with tensors needed for backward
        """
        TR = T * R

        # Compute exp(cs) and exp(-cs) directly from the cumsum.
        # The cumsum gives cs_t = sum_{i=0}^{t} ADT[i]. Since both Q and K are
        # scaled, any constant offset cancels: (exp(cs_t)*Q) @ (exp(-cs_s)*K)^T
        # = exp(cs_t - cs_s) * Q @ K^T, which is exactly L[t,s] * Q@K^T.
        # Use reciprocal for exp_neg_cs to avoid overflow: exp(-cs) can grow
        # exponentially for long sequences, but 1/exp(cs) is bounded by the
        # smallest exp(cs) value that's still representable in bf16.
        exp_cs = ttnn.typecast(ttnn.exp(ADT_cumsum), ttnn.bfloat16)      # (B,H,T)
        exp_neg_cs = ttnn.reciprocal(exp_cs)  # (B,H,T) — safe: 1/exp(cs)
        # Reshape to (B,H,T,1) for backward compatibility (backward does expand to (B,H,T,R))
        exp_cs = ttnn.reshape(exp_cs, [B, H, T, 1])
        exp_neg_cs = ttnn.reshape(exp_neg_cs, [B, H, T, 1])

        # Fold R into T: (B,H,R,T,N) -> (B,H,T,R,N) -> (B,H,T*R,N)
        Q = ttnn.permute(Q_perm, [0, 1, 3, 2, 4])  # (B,H,T,R,N)
        Q = ttnn.reshape(Q, [B, H, TR, N])
        K = ttnn.permute(K_perm, [0, 1, 3, 2, 4])  # (B,H,T,R,N)
        K = ttnn.reshape(K, [B, H, TR, N])
        V = ttnn.permute(V_perm, [0, 1, 3, 2, 4])  # (B,H,T,R,P)
        V = ttnn.reshape(V, [B, H, TR, P])

        # Scale Q and K by the decay factors
        # exp_cs: (B,H,T,1) -> expand to (B,H,T,R) -> reshape to (B,H,T*R,1)
        exp_cs_exp = ttnn.reshape(
            ttnn.expand(exp_cs, [B, H, T, R]), [B, H, TR, 1])
        exp_neg_cs_exp = ttnn.reshape(
            ttnn.expand(exp_neg_cs, [B, H, T, R]), [B, H, TR, 1])

        Q_scaled = ttnn.mul(Q, exp_cs_exp)   # (B,H,T*R,N)
        K_scaled = ttnn.mul(K, exp_neg_cs_exp)  # (B,H,T*R,N)

        # QK = Q_scaled @ K_scaled^T = (B,H,T*R,T*R)
        QK = ttnn.matmul(Q_scaled, ttnn.transpose(K_scaled, -2, -1),
                         compute_kernel_config=_HIFI_FP32_ACC)

        # Apply strict causal mask (cached per T,R)
        QK_masked = ttnn.mul(QK, self._get_strict_mask_exp(T, R, device))

        # Y = QK_masked @ V = (B,H,T*R,P)
        Y = ttnn.matmul(QK_masked, V,
                       compute_kernel_config=_HIFI_FP32_ACC)

        # Reshape: (B,H,T*R,P) -> (B,H,T,R,P) -> permute to (B,H,R,T,P)
        Y = ttnn.reshape(Y, [B, H, T, R, P])
        out_attn = ttnn.permute(Y, [0, 1, 3, 2, 4])  # (B,H,R,T,P)

        # For backward, we need the raw QK (before masking) and the masked version
        # Also need Q, K, V in flat form, and the scaling factors
        cache = {
            "Q_flat": Q, "K_flat": K, "V_flat": V,
            "Q_scaled": Q_scaled, "K_scaled_attn": K_scaled,
            "QK": QK, "QK_masked": QK_masked,
            "exp_cs": exp_cs, "exp_neg_cs": exp_neg_cs,
            "exp_cs_exp": exp_cs_exp, "exp_neg_cs_exp": exp_neg_cs_exp,
            "strict_mask_exp": self._get_strict_mask_exp(T, R, device),
        }
        return out_attn, cache

    def _full_tt_attention_backward(self, grad_out_attn, c, B, H, R, T, N, P, device):
        """Backward of full T×T MIMO attention.

        Forward: Q_scaled = Q * exp_cs, K_scaled = K * exp_neg_cs
                 QK = Q_scaled @ K_scaled^T
                 QK_masked = QK * strict_mask
                 Y = QK_masked @ V

        Backward: standard matmul backward through each step.

        Returns: (grad_Q_rot, grad_K_scaled, grad_V_proj_attn, grad_L_accum, grad_ADT_from_chunk)
        """
        chunk = c["chunk_cache"]
        Q = chunk["Q_flat"]           # (B,H,T*R,N) — unscaled
        K = chunk["K_flat"]           # (B,H,T*R,N) — unscaled
        V = chunk["V_flat"]           # (B,H,T*R,P)
        Q_scaled = chunk["Q_scaled"]  # (B,H,T*R,N) — Q * exp_cs
        K_scaled = chunk["K_scaled_attn"]  # (B,H,T*R,N) — K * exp_neg_cs
        QK = chunk["QK"]              # (B,H,T*R,T*R) — raw (before mask)
        QK_masked = chunk["QK_masked"]  # (B,H,T*R,T*R) — after mask
        exp_cs_exp = chunk["exp_cs_exp"]      # (B,H,T*R,1) — cached from forward
        exp_neg_cs_exp = chunk["exp_neg_cs_exp"]  # (B,H,T*R,1) — cached from forward
        strict_mask_exp = chunk["strict_mask_exp"]  # (1,1,1,T*R,T*R)

        TR = T * R

        # grad_out_attn: (B,H,R,T,P) -> (B,H,T,R,P) -> (B,H,T*R,P)
        grad_Y = ttnn.permute(grad_out_attn, [0, 1, 3, 2, 4])  # (B,H,T,R,P)
        grad_Y = ttnn.reshape(grad_Y, [B, H, TR, P])

        # Backward of Y = QK_masked @ V
        grad_QK_masked = ttnn.matmul(grad_Y, ttnn.transpose(V, -2, -1),
                                    compute_kernel_config=_HIFI_FP32_ACC)
        grad_V = ttnn.matmul(ttnn.transpose(QK_masked, -2, -1), grad_Y,
                            compute_kernel_config=_HIFI_FP32_ACC)

        # Backward of QK_masked = QK * strict_mask_exp
        grad_QK = ttnn.mul(grad_QK_masked, strict_mask_exp)

        # Backward of QK = Q_scaled @ K_scaled^T
        grad_Q_scaled = ttnn.matmul(grad_QK, K_scaled,
                                   compute_kernel_config=_HIFI_FP32_ACC)
        grad_K_scaled = ttnn.matmul(ttnn.transpose(grad_QK, -2, -1), Q_scaled,
                                   compute_kernel_config=_HIFI_FP32_ACC)

        # Backward of Q_scaled = Q * exp_cs_exp (use cached exp_cs_exp)
        grad_Q = ttnn.mul(grad_Q_scaled, exp_cs_exp)
        grad_K = ttnn.mul(grad_K_scaled, exp_neg_cs_exp)

        # Reshape gradients back to (B,T,R,H,N) / (B,T,R,H,P)
        grad_Q = ttnn.reshape(grad_Q, [B, H, T, R, N])
        grad_Q = ttnn.permute(grad_Q, [0, 1, 3, 2, 4])  # (B,H,R,T,N)
        grad_Q_rot = ttnn.permute(grad_Q, [0, 3, 2, 1, 4])  # (B,T,R,H,N)

        grad_K = ttnn.reshape(grad_K, [B, H, T, R, N])
        grad_K = ttnn.permute(grad_K, [0, 1, 3, 2, 4])
        grad_K_scaled = ttnn.permute(grad_K, [0, 3, 2, 1, 4])

        grad_V = ttnn.reshape(grad_V, [B, H, T, R, P])
        grad_V = ttnn.permute(grad_V, [0, 1, 3, 2, 4])
        grad_V_proj_attn = ttnn.permute(grad_V, [0, 3, 2, 1, 4])

        # grad_L_accum: grad_QK_masked * QK, then sum over R blocks
        grad_L_raw = ttnn.mul(grad_QK_masked, QK)  # (B,H,T*R,T*R)
        grad_L_5d = ttnn.reshape(grad_L_raw, [B, H, T, R, T, R])
        grad_L_4d = ttnn.sum(grad_L_5d, dim=3)  # (B,H,T,T,R)
        grad_L_summed = ttnn.sum(grad_L_4d, dim=4)  # (B,H,T,T)
        # Apply strict causal mask (cached per T)
        grad_L_accum = ttnn.mul(grad_L_summed, self._get_strict_mask_4d(T, device))
        grad_L_accum = ttnn.reshape(grad_L_accum, [B, H, 1, T, T])

        grad_ADT_from_chunk = None
        return grad_Q_rot, grad_K_scaled, grad_V_proj_attn, grad_L_accum, grad_ADT_from_chunk

    def _chunked_attention(self, Q_perm, K_perm, V_perm, ADT, B, H, R, T, N, P, device):
        """Chunked MIMO attention — replaces the R² loop.

        Splits the sequence into chunks of size C (C = 64//R).
        Intrachunk: single batched matmul per chunk (fused R*C × R*C).
        Interchunk: sequential state propagation (nchunks iterations).

        Returns: (out_attn, cache_dict) where out_attn is (B,H,R,T,P) strictly causal
                 (excluding the diagonal t==s terms).
        """
        C = max(1, min(64 // R, T))  # chunk_size, capped at T
        assert T % C == 0, f"T={T} must be divisible by chunk_size C={C}"
        nchunks = T // C
        fused = C * R

        # Reshape (B,H,R,T,N) -> (B,H,nchunks,fused,N)
        # fused index = cs * R + r (matching official kernel)
        Q_chunk = ttnn.reshape(Q_perm, [B, H, R, nchunks, C, N])
        Q_chunk = ttnn.permute(Q_chunk, [0, 1, 3, 4, 2, 5])  # (B,H,nchunks,C,R,N)
        Q_chunk = ttnn.reshape(Q_chunk, [B, H, nchunks, fused, N])

        K_chunk = ttnn.reshape(K_perm, [B, H, R, nchunks, C, N])
        K_chunk = ttnn.permute(K_chunk, [0, 1, 3, 4, 2, 5])
        K_chunk = ttnn.reshape(K_chunk, [B, H, nchunks, fused, N])

        V_chunk = ttnn.reshape(V_perm, [B, H, R, nchunks, C, P])
        V_chunk = ttnn.permute(V_chunk, [0, 1, 3, 4, 2, 5])
        V_chunk = ttnn.reshape(V_chunk, [B, H, nchunks, fused, P])

        # Per-chunk cumsum of ADT
        ADT_4d = ttnn.reshape(ADT, [B, H, nchunks, C])  # bf16
        ADT_4d_32 = ttnn.typecast(ADT_4d, ttnn.float32)
        DA_CS_chunk = ttnn.cumsum(ADT_4d_32, dim=-1)  # (B,H,nchunks,C) fp32, per-chunk inclusive
        DA_CS_sum = ttnn.sum(ADT_4d_32, dim=-1)  # (B,H,nchunks) fp32
        DA_CS_REV = ttnn.sub(ttnn.reshape(DA_CS_sum, [B, H, nchunks, 1]), DA_CS_chunk)  # (B,H,nchunks,C) fp32

        # Per-chunk segsum: segsum[i,j] = DA_CS[i] - DA_CS[j] for i > j, 0 for i <= j
        segsum = ttnn.sub(
            ttnn.reshape(DA_CS_chunk, [B, H, nchunks, C, 1]),
            ttnn.reshape(DA_CS_chunk, [B, H, nchunks, 1, C])
        )  # (B,H,nchunks,C,C) fp32
        # Strictly lower triangular mask (i > j)
        segsum = ttnn.where(self._strict_causal_mask(C, ttnn.float32), segsum,
                            ttnn.zeros_like(segsum))
        decay_intra = ttnn.exp(segsum)  # (B,H,nchunks,C,C) fp32

        # Expand decay to fused: (B,H,nchunks,C,C) -> (B,H,nchunks,fused,fused)
        decay_fused = ttnn.reshape(decay_intra, [B, H, nchunks, C, 1, C, 1])
        decay_fused = ttnn.expand(decay_fused, [B, H, nchunks, C, R, C, R])
        decay_fused = ttnn.reshape(decay_fused, [B, H, nchunks, fused, fused])

        # Intrachunk QK: (B,H,nchunks,fused,N) @ (B,H,nchunks,N,fused) -> (B,H,nchunks,fused,fused)
        QK_intra = ttnn.matmul(Q_chunk, ttnn.transpose(K_chunk, -2, -1),
                               compute_kernel_config=_HIFI_FP32_ACC)

        # Apply strictly-causal mask (cs_i > cs_j) and decay
        mask_fused = self._strict_causal_fused_mask(C, R, ttnn.float32)  # (fused,fused)
        QK_intra = ttnn.mul(QK_intra, mask_fused)  # zero out non-causal
        QK_intra = ttnn.mul(QK_intra, decay_fused)  # apply decay

        # Intrachunk output: (B,H,nchunks,fused,fused) @ (B,H,nchunks,fused,P) -> (B,H,nchunks,fused,P)
        o_intra = ttnn.matmul(QK_intra, V_chunk)

        # Interchunk: sequential state propagation
        # K_state = K_chunk * exp(DA_CS_REV), expanded over R
        exp_DA_CS_REV = ttnn.exp(DA_CS_REV)  # (B,H,nchunks,C) fp32
        exp_DA_CS_REV_fused = ttnn.reshape(exp_DA_CS_REV, [B, H, nchunks, C, 1])
        exp_DA_CS_REV_fused = ttnn.expand(exp_DA_CS_REV_fused, [B, H, nchunks, C, R])
        exp_DA_CS_REV_fused = ttnn.reshape(exp_DA_CS_REV_fused, [B, H, nchunks, fused])
        exp_DA_CS_REV_fused = ttnn.typecast(exp_DA_CS_REV_fused, ttnn.bfloat16)
        K_state = ttnn.mul(K_chunk, ttnn.reshape(exp_DA_CS_REV_fused, [B, H, nchunks, fused, 1]))

        # B_state = K_state^T @ V_chunk — (B,H,nchunks,N,P)
        B_state = ttnn.matmul(ttnn.transpose(K_state, -2, -1), V_chunk,
                              compute_kernel_config=_HIFI_FP32_ACC)

        # Sequential loop over chunks
        exp_DA_CS = ttnn.typecast(ttnn.exp(DA_CS_chunk), ttnn.bfloat16)  # (B,H,nchunks,C)
        exp_DA_CS_sum = ttnn.typecast(ttnn.exp(DA_CS_sum), ttnn.bfloat16)  # (B,H,nchunks)

        # Precompute per-chunk slices to reduce ops in the loop
        Q_slices = [ttnn.reshape(
            ttnn.slice(Q_chunk, [0, 0, i, 0, 0], [B, H, i + 1, fused, N]),
            [B, H, fused, N]) for i in range(nchunks)]
        exp_cs_slices = [ttnn.reshape(
            ttnn.slice(exp_DA_CS, [0, 0, i, 0], [B, H, i + 1, C]),
            [B, H, C, 1]) for i in range(nchunks)]
        exp_cs_slices = [ttnn.reshape(
            ttnn.expand(s, [B, H, C, R]), [B, H, fused, 1]) for s in exp_cs_slices]
        # Pre-scale Q by exp_cs to eliminate per-iteration mul
        Q_scaled_slices = [ttnn.mul(Q_slices[i], exp_cs_slices[i]) for i in range(nchunks)]
        exp_sum_slices = [ttnn.reshape(
            ttnn.slice(exp_DA_CS_sum, [0, 0, i], [B, H, i + 1]),
            [B, H, 1, 1]) for i in range(nchunks)]
        B_slices = [ttnn.reshape(
            ttnn.slice(B_state, [0, 0, i, 0, 0], [B, H, i + 1, N, P]),
            [B, H, N, P]) for i in range(nchunks)]

        states = ttnn.zeros((B, H, N, P), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device)
        states_list = [states]  # states before each chunk's update
        o_inter_list = []
        for i in range(nchunks):
            # o_inter = (Q_i * exp_cs) @ states — pre-scaled Q eliminates per-iter mul
            o_inter = ttnn.matmul(Q_scaled_slices[i], states,
                                  compute_kernel_config=_HIFI_FP32_ACC)
            o_inter_list.append(o_inter)

            # State update: states = exp_sum * states + B_i (fused via mac)
            states = ttnn.mac(exp_sum_slices[i], states, B_slices[i])
            states_list.append(states)

        # Stack o_inter: (B,H,nchunks,fused,P)
        o_inter = ttnn.concat(o_inter_list, dim=2)
        o_inter = ttnn.reshape(o_inter, [B, H, nchunks, fused, P])

        # Combine intrachunk + interchunk
        o_chunk = ttnn.add(o_intra, o_inter)  # (B,H,nchunks,fused,P)

        # Reshape back to (B,H,R,T,P)
        o_chunk = ttnn.reshape(o_chunk, [B, H, nchunks, C, R, P])
        o_chunk = ttnn.permute(o_chunk, [0, 1, 4, 2, 3, 5])  # (B,H,R,nchunks,C,P)
        out_attn = ttnn.reshape(o_chunk, [B, H, R, T, P])

        cache = {
            "Q_chunk": Q_chunk, "K_chunk": K_chunk, "V_chunk": V_chunk,
            "K_state": K_state, "B_state": B_state,
            "DA_CS_chunk": DA_CS_chunk, "DA_CS_sum": DA_CS_sum,
            "DA_CS_REV": DA_CS_REV, "exp_DA_CS_REV_fused": exp_DA_CS_REV_fused,
            "decay_intra": decay_intra,
            "decay_fused": decay_fused, "mask_fused": mask_fused,
            "exp_DA_CS": exp_DA_CS, "exp_DA_CS_sum": exp_DA_CS_sum,
            "QK_intra": QK_intra, "o_intra": o_intra, "o_inter": o_inter,
            "chunk_size": C, "nchunks": nchunks, "fused": fused,
            "states_final": states, "states_list": states_list,
            "Q_scaled_slices": Q_scaled_slices,
        }
        return out_attn, cache

    def _chunked_attention_backward(self, grad_out_attn, c, B, H, R, T, N, P, device):
        """Backward of the chunked attention using chunked structure.

        Intrachunk: standard matmul backward of o_intra = QK_masked @ V_chunk
        Interchunk: reverse sequential loop for state propagation backward

        Returns: (grad_Q_rot, grad_K_scaled, grad_V_proj_attn, grad_L_accum, grad_ADT_from_chunk)
        """
        chunk = c["chunk_cache"]
        C = chunk["chunk_size"]
        nchunks = chunk["nchunks"]
        fused = chunk["fused"]

        Q_chunk = chunk["Q_chunk"]  # (B,H,nchunks,fused,N)
        K_chunk = chunk["K_chunk"]  # (B,H,nchunks,fused,N)
        V_chunk = chunk["V_chunk"]  # (B,H,nchunks,fused,P)
        K_state = chunk["K_state"]  # (B,H,nchunks,N,P)
        B_state = chunk["B_state"]  # (B,H,nchunks,N,P)
        decay_intra = chunk["decay_intra"]  # (B,H,nchunks,fused,fused)
        mask_fused = chunk["mask_fused"]  # (1,1,1,fused,fused)
        decay_fused = chunk["decay_fused"]  # (B,H,nchunks,fused,fused)
        QK_intra = chunk["QK_intra"]  # (B,H,nchunks,fused,fused) — masked+decayed
        exp_DA_CS = chunk["exp_DA_CS"]  # (B,H,nchunks,C)
        exp_DA_CS_sum = chunk["exp_DA_CS_sum"]  # (B,H,nchunks)
        DA_CS_REV = chunk["exp_DA_CS_REV_fused"]  # (B,H,nchunks,fused) — exp'd, fused
        o_intra = chunk["o_intra"]  # (B,H,nchunks,fused,P)
        o_inter = chunk["o_inter"]  # (B,H,nchunks,fused,P)

        # grad_out_attn is (B,H,R,T,P) — reshape to (B,H,nchunks,fused,P)
        grad_o = ttnn.reshape(grad_out_attn, [B, H, R, nchunks, C, P])
        grad_o = ttnn.permute(grad_o, [0, 1, 3, 4, 2, 5])  # (B,H,nchunks,C,R,P)
        grad_o = ttnn.reshape(grad_o, [B, H, nchunks, fused, P])

        # --- Intrachunk backward ---
        # o_intra = QK_intra @ V_chunk  (QK_intra already has mask*decay applied)
        grad_QK_intra = ttnn.matmul(grad_o, ttnn.transpose(V_chunk, -2, -1),
                                   compute_kernel_config=_HIFI_FP32_ACC)  # (B,H,nchunks,fused,fused)
        grad_V_from_intra = ttnn.matmul(ttnn.transpose(QK_intra, -2, -1), grad_o,
                                       compute_kernel_config=_HIFI_FP32_ACC)  # (B,H,nchunks,fused,P)

        # grad_QK_raw = grad_QK_intra * decay * mask (chain through element-wise)
        grad_QK_raw_intra = ttnn.mul(grad_QK_intra, ttnn.mul(decay_fused, mask_fused))

        # grad_Q_intra = grad_QK_raw @ K_chunk
        grad_Q_intra = ttnn.matmul(grad_QK_raw_intra, K_chunk,
                                  compute_kernel_config=_HIFI_FP32_ACC)  # (B,H,nchunks,fused,N)
        # grad_K_intra = grad_QK_raw^T @ Q_chunk
        grad_K_intra = ttnn.matmul(ttnn.transpose(grad_QK_raw_intra, -2, -1), Q_chunk,
                                  compute_kernel_config=_HIFI_FP32_ACC)  # (B,H,nchunks,fused,N)

        # grad_decay_intra = grad_QK_intra * QK_intra_raw (before mask/decay)
        # QK_intra_raw = Q_chunk @ K_chunk^T (recompute)
        QK_intra_raw = ttnn.matmul(Q_chunk, ttnn.transpose(K_chunk, -2, -1),
                                  compute_kernel_config=_HIFI_FP32_ACC)  # (B,H,nchunks,fused,fused)
        grad_decay_intra = ttnn.mul(grad_QK_intra, QK_intra_raw)  # (B,H,nchunks,fused,fused)
        # Apply mask (zero out diagonal and upper triangle)
        grad_decay_intra = ttnn.mul(grad_decay_intra, mask_fused)

        # --- Interchunk backward ---
        # Forward: o_inter[i] = exp_cs[i] * Q_i @ states_i
        #          states_{i+1} = exp_sum[i] * states_i + B_i
        # B_i = K_state_i^T @ V_chunk_i  (K_state_i = K_chunk_i * DA_CS_REV_i)
        #
        # Backward: reverse loop
        # grad_states_i gets gradient from o_inter[i] AND from states_{i+1}

        # Precompute per-chunk slices for the reverse loop
        Q_slices = [ttnn.reshape(
            ttnn.slice(Q_chunk, [0, 0, i, 0, 0], [B, H, i + 1, fused, N]),
            [B, H, fused, N]) for i in range(nchunks)]
        V_slices = [ttnn.reshape(
            ttnn.slice(V_chunk, [0, 0, i, 0, 0], [B, H, i + 1, fused, P]),
            [B, H, fused, P]) for i in range(nchunks)]
        K_state_slices = [ttnn.reshape(
            ttnn.slice(K_state, [0, 0, i, 0, 0], [B, H, i + 1, fused, N]),
            [B, H, fused, N]) for i in range(nchunks)]
        B_state_slices = [ttnn.reshape(
            ttnn.slice(B_state, [0, 0, i, 0, 0], [B, H, i + 1, N, P]),
            [B, H, N, P]) for i in range(nchunks)]
        K_chunk_slices = [ttnn.reshape(
            ttnn.slice(K_chunk, [0, 0, i, 0, 0], [B, H, i + 1, fused, N]),
            [B, H, fused, N]) for i in range(nchunks)]
        DA_CS_REV_slices = [ttnn.reshape(
            ttnn.slice(DA_CS_REV, [0, 0, i, 0], [B, H, i + 1, fused]),
            [B, H, fused, 1]) for i in range(nchunks)]
        exp_cs_slices = [ttnn.reshape(
            ttnn.slice(exp_DA_CS, [0, 0, i, 0], [B, H, i + 1, C]),
            [B, H, C, 1]) for i in range(nchunks)]
        exp_cs_slices = [ttnn.reshape(
            ttnn.expand(s, [B, H, C, R]), [B, H, fused, 1]) for s in exp_cs_slices]
        exp_sum_slices = [ttnn.reshape(
            ttnn.slice(exp_DA_CS_sum, [0, 0, i], [B, H, i + 1]),
            [B, H, 1, 1]) for i in range(nchunks)]

        grad_o_inter = grad_o  # gradient of o_inter = gradient of out_attn

        # Use cached forward states (no recompute needed)
        states_list = chunk["states_list"]

        # Precompute grad_o slices and pre-scale by exp_cs outside the loop
        grad_o_slices = [ttnn.reshape(
            ttnn.slice(grad_o_inter, [0, 0, i, 0, 0], [B, H, i + 1, fused, P]),
            [B, H, fused, P]) for i in range(nchunks)]
        grad_o_scaled_slices = [ttnn.mul(grad_o_slices[i], exp_cs_slices[i])
                                for i in range(nchunks)]

        # Accumulators
        grad_Q_inter_list = [None] * nchunks
        grad_K_inter_list = [None] * nchunks
        grad_V_inter_list = [None] * nchunks

        grad_states = ttnn.zeros((B, H, N, P), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device)

        for i in range(nchunks - 1, -1, -1):
            states_i = states_list[i]
            grad_o_scaled = grad_o_scaled_slices[i]

            # grad_Q_inter[i] = grad_o_scaled @ states_i^T
            grad_Q_i = ttnn.matmul(grad_o_scaled, ttnn.transpose(states_i, -2, -1),
                                  compute_kernel_config=_HIFI_FP32_ACC)
            grad_Q_inter_list[i] = grad_Q_i

            # grad_B_state[i] = grad_states_{i+1}
            grad_B_i = grad_states

            # grad_states_i = exp_sum * grad_states + Q^T @ grad_o_scaled (fused via mac)
            grad_states_from_o = ttnn.matmul(ttnn.transpose(Q_slices[i], -2, -1), grad_o_scaled,
                                             compute_kernel_config=_HIFI_FP32_ACC)
            grad_states = ttnn.mac(exp_sum_slices[i], grad_states, grad_states_from_o)

            # B_i = K_state_i^T @ V_chunk_i
            # grad_K_state_i = V_chunk_i @ grad_B_i^T
            # grad_V_chunk_i_from_inter = K_state_i @ grad_B_i
            grad_K_state_i = ttnn.matmul(V_slices[i], ttnn.transpose(grad_B_i, -2, -1))
            grad_V_i_from_inter = ttnn.matmul(K_state_slices[i], grad_B_i)
            grad_V_inter_list[i] = grad_V_i_from_inter

            # K_state_i = K_chunk_i * DA_CS_REV_i
            # grad_K_chunk_i_from_inter = grad_K_state_i * DA_CS_REV_i
            # grad_DA_CS_REV_i = grad_K_state_i * K_chunk_i — feeds to ADT
            grad_K_chunk_i = ttnn.mul(grad_K_state_i, DA_CS_REV_slices[i])
            grad_K_inter_list[i] = grad_K_chunk_i

        # Stack interchunk gradients
        grad_Q_inter = ttnn.concat(
            [ttnn.reshape(g, [B, H, 1, fused, N]) for g in grad_Q_inter_list], dim=2)
        grad_K_inter = ttnn.concat(
            [ttnn.reshape(g, [B, H, 1, fused, N]) for g in grad_K_inter_list], dim=2)
        grad_V_inter = ttnn.concat(
            [ttnn.reshape(g, [B, H, 1, fused, P]) for g in grad_V_inter_list], dim=2)

        # --- Combine intrachunk + interchunk ---
        grad_Q_chunk = ttnn.add(grad_Q_intra, grad_Q_inter)  # (B,H,nchunks,fused,N)
        grad_K_chunk = ttnn.add(grad_K_intra, grad_K_inter)  # (B,H,nchunks,fused,N)
        grad_V_chunk = ttnn.add(grad_V_from_intra, grad_V_inter)  # (B,H,nchunks,fused,P)

        # --- Reshape back to (B,T,R,H,N/P) — matching old R² loop format ---
        # fused index = cs * R + r -> need to unpermute
        grad_Q_chunk = ttnn.reshape(grad_Q_chunk, [B, H, nchunks, C, R, N])
        grad_Q_chunk = ttnn.permute(grad_Q_chunk, [0, 1, 4, 2, 3, 5])  # (B,H,R,nchunks,C,N)
        grad_Q_chunk = ttnn.reshape(grad_Q_chunk, [B, H, R, T, N])
        grad_Q_rot = ttnn.permute(grad_Q_chunk, [0, 3, 2, 1, 4])  # (B,T,R,H,N)

        grad_K_chunk = ttnn.reshape(grad_K_chunk, [B, H, nchunks, C, R, N])
        grad_K_chunk = ttnn.permute(grad_K_chunk, [0, 1, 4, 2, 3, 5])
        grad_K_chunk = ttnn.reshape(grad_K_chunk, [B, H, R, T, N])
        grad_K_scaled = ttnn.permute(grad_K_chunk, [0, 3, 2, 1, 4])  # (B,T,R,H,N)

        grad_V_chunk = ttnn.reshape(grad_V_chunk, [B, H, nchunks, C, R, P])
        grad_V_chunk = ttnn.permute(grad_V_chunk, [0, 1, 4, 2, 3, 5])
        grad_V_chunk = ttnn.reshape(grad_V_chunk, [B, H, R, T, P])
        grad_V_proj_attn = ttnn.permute(grad_V_chunk, [0, 3, 2, 1, 4])  # (B,T,R,H,P)

        # --- grad_L_accum via batched matmul (replaces R² loop) ---
        # grad_L[t,s] = mask[t,s] * sum_{r_q, r_k} (grad_out@V^T)[r_q,r_k,t,s] * (Q@K^T)[r_q,r_k,t,s]
        # Compute Q@K^T and grad_out@V^T as single (B,H,R*T,R*T) matmuls,
        # multiply, then sum over R blocks to get (B,H,1,T,T).
        Q_perm = c["Q_perm"]  # (B,H,R,T,N)
        K_perm = c["K_perm"]
        V_perm = c["V_perm"]

        Q_flat = ttnn.reshape(Q_perm, [B, H, R * T, N])
        K_flat = ttnn.reshape(K_perm, [B, H, R * T, N])
        QK_full = ttnn.matmul(Q_flat, ttnn.transpose(K_flat, -2, -1),
                            compute_kernel_config=_HIFI_FP32_ACC)  # (B,H,R*T,R*T)

        grad_flat = ttnn.reshape(grad_out_attn, [B, H, R * T, P])
        V_flat = ttnn.reshape(V_perm, [B, H, R * T, P])
        gQK_full = ttnn.matmul(grad_flat, ttnn.transpose(V_flat, -2, -1),
                              compute_kernel_config=_HIFI_FP32_ACC)  # (B,H,R*T,R*T)

        grad_L_raw = ttnn.mul(gQK_full, QK_full)  # (B,H,R*T,R*T)
        # Sum over R blocks: reshape to separate R from T, sum over R dims
        # (B,H,R*T,R*T) -> (B,H,R,T,R*T) -> sum dim 2 -> (B,H,T,R*T)
        # -> (B,H,T,R,T) -> sum dim 3 -> (B,H,T,T)
        grad_L_5d = ttnn.reshape(grad_L_raw, [B, H, R, T, R * T])
        grad_L_4d = ttnn.sum(grad_L_5d, dim=2)  # (B,H,T,R*T)
        grad_L_4d = ttnn.reshape(grad_L_4d, [B, H, T, R, T])
        grad_L_summed = ttnn.sum(grad_L_4d, dim=3)  # (B,H,T,T)
        # Apply strict causal mask
        strict_mask = self._strict_causal_full_mask(T, ttnn.bfloat16)  # (1,1,1,T,T)
        strict_mask_4d = ttnn.reshape(strict_mask, [1, 1, T, T])
        grad_L_accum = ttnn.mul(grad_L_summed, strict_mask_4d)
        # Reshape to 5D for downstream: (B,H,1,T,T)
        grad_L_accum = ttnn.reshape(grad_L_accum, [B, H, 1, T, T])

        grad_ADT_from_chunk = None
        return grad_Q_rot, grad_K_scaled, grad_V_proj_attn, grad_L_accum, grad_ADT_from_chunk

    def _strict_causal_mask(self, C, dtype=ttnn.float32):
        """(C,C) strictly lower triangular mask (1 for i > j, 0 for i <= j). Cached."""
        cache_key = ("strict_causal", C, dtype)
        if cache_key in self._cache:
            return self._cache[cache_key]
        mask = torch.tril(torch.ones(C, C, dtype=torch.float32), diagonal=-1)
        result = ttnn.from_torch(mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        self._cache[cache_key] = result
        return result

    def _strict_causal_full_mask(self, T, dtype=ttnn.bfloat16):
        """(1,1,1,T,T) strictly causal mask (1 for s < t, 0 for s >= t). Cached."""
        cache_key = ("strict_causal_full", T, dtype)
        if cache_key in self._cache:
            return self._cache[cache_key]
        mask = torch.tril(torch.ones(T, T, dtype=torch.float32), diagonal=-1)
        mask = mask.reshape(1, 1, 1, T, T)
        result = ttnn.from_torch(mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        self._cache[cache_key] = result
        return result

    def _strict_causal_fused_mask(self, C, R, dtype=ttnn.float32):
        """(fused,fused) mask: 1 for cs_i > cs_j, 0 otherwise. fused = C*R. Cached."""
        fused = C * R
        cache_key = ("strict_causal_fused", C, R, dtype)
        if cache_key in self._cache:
            return self._cache[cache_key]
        mask_C = torch.tril(torch.ones(C, C, dtype=torch.float32), diagonal=-1)
        # Expand to (C,R,C,R) -> (fused,fused)
        mask = mask_C.unsqueeze(1).unsqueeze(-1).expand(C, R, C, R).reshape(fused, fused)
        result = ttnn.from_torch(mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        # Reshape to (1,1,1,fused,fused) for broadcasting
        result = ttnn.reshape(result, [1, 1, 1, fused, fused])
        self._cache[cache_key] = result
        return result

    # -- Forward --

    def forward(self, x):
        """x: (B, T, d_model) -> returns (B, T, d_model)"""
        B, T, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        H, R, N, P = self.nheads, self.R, self.d_state, self.headdim
        device = self.device
        n_angles = self.num_rope_angles
        ngroups = self.ngroups

        # --- in_proj ---
        _t0 = _t("fwd_in_proj")
        xzBCdtAtrap = ttnn.linear(x, self.in_proj_weight)

        # Split: [z, x, B, C, dt, A, trap, angles]
        off = 0
        z_raw = ttnn.slice(xzBCdtAtrap, [0, 0, off], [B, T, off + self.d_inner]); off += self.d_inner
        x_raw = ttnn.slice(xzBCdtAtrap, [0, 0, off], [B, T, off + self.d_inner]); off += self.d_inner
        B_raw = ttnn.slice(xzBCdtAtrap, [0, 0, off], [B, T, off + self.d_state * ngroups * R]); off += self.d_state * ngroups * R
        C_raw = ttnn.slice(xzBCdtAtrap, [0, 0, off], [B, T, off + self.d_state * ngroups * R]); off += self.d_state * ngroups * R
        dt_raw = ttnn.slice(xzBCdtAtrap, [0, 0, off], [B, T, off + H]); off += H
        A_raw = ttnn.slice(xzBCdtAtrap, [0, 0, off], [B, T, off + H]); off += H
        trap_raw = ttnn.slice(xzBCdtAtrap, [0, 0, off], [B, T, off + H]); off += H
        angles_raw = ttnn.slice(xzBCdtAtrap, [0, 0, off], [B, T, off + n_angles])
        _t("fwd_in_proj", _t0)

        # --- Reshape z, x (V) to (B, T, H, P) ---
        # Keep the (B, T, H, P) layout: the only consumers are the MIMO
        # up-projections below, which broadcast over R as (B, T, 1, H, P).
        _t1 = _t("fwd_reshape_bc")
        z = ttnn.reshape(z_raw, [B, T, H, P])
        V = ttnn.reshape(x_raw, [B, T, H, P])

        # --- Reshape B, C -> (B, T, R, H, N) with GQA expansion ---
        B_mat = ttnn.reshape(B_raw, [B, T, R, ngroups, N])
        C_mat = ttnn.reshape(C_raw, [B, T, R, ngroups, N])
        if ngroups < H:
            B_mat = ttnn.expand(B_mat, [B, T, R, H, N])
            C_mat = ttnn.expand(C_mat, [B, T, R, H, N])

        # --- QKNorm (5D input, no reshape to 3D) ---
        B_normed = self.B_norm.forward(B_mat)
        C_normed = self.C_norm.forward(C_mat)

        # --- Add B/C biases -> K, Q (pre-rotation) ---
        # Pre-computed (1,1,R,H,N) bf16 bases broadcast against (B,T,R,H,N)
        K = ttnn.add(B_normed, self._B_bias_base)
        Q = ttnn.add(C_normed, self._C_bias_base)
        _t("fwd_norm_bias", _t1)

        # --- DT, A, ADT ---
        _t2 = _t("fwd_dt_a")
        dt_bias_bf = ttnn.typecast(ttnn.reshape(self.dt_bias, [1, 1, H]), ttnn.bfloat16)
        DT = ttnn.softplus(ttnn.add(dt_raw, dt_bias_bf))
        A_log_bf = ttnn.typecast(ttnn.reshape(self.A_log, [1, 1, H]), ttnn.bfloat16)
        A = ttnn.neg(ttnn.softplus(ttnn.add(A_raw, A_log_bf)))
        ADT = ttnn.permute(ttnn.mul(A, DT), [0, 2, 1])  # (B, H, T)

        # Pre-compute fp32 derivatives for backward (saves ~10 ops in bwd_dt_trap_qk)
        _DT32 = ttnn.typecast(DT, ttnn.float32)
        _A32 = ttnn.typecast(A, ttnn.float32)
        _ones32_bth = self._ones_tensor((B, T, H), dtype=ttnn.float32)
        # dA/d(A_raw+A_log) = -(1 - exp(A)) = exp(A) - 1
        _a_deriv32 = ttnn.neg(ttnn.sub(_ones32_bth, ttnn.exp(_A32)))
        # dDT/d(dt_raw+dt_bias) = 1 - exp(-DT) = sigmoid(dt_raw+dt_bias)
        _dt_deriv32 = ttnn.sub(_ones32_bth, ttnn.exp(ttnn.neg(_DT32)))
        _t("fwd_dt_a", _t2)

        # --- Angles and RoPE ---
        _t3 = _t("fwd_rope")
        angles = ttnn.expand(ttnn.reshape(angles_raw, [B, T, 1, n_angles]), [B, T, H, n_angles])
        angles_proc = ttnn.mul(ttnn.tanh(angles), self._pi_tt)
        DT_exp = ttnn.reshape(DT, [B, T, H, 1])
        angles_scaled = ttnn.mul(angles_proc, DT_exp)
        # Accumulate the RoPE phase in fp32: this cumsum runs the length of the
        # sequence and feeds cos/sin, so bf16 drift shows up as a phase error
        # that grows with T.
        angles_cumsum32 = ttnn.cumsum(ttnn.typecast(angles_scaled, ttnn.float32), dim=1)
        cos_a = ttnn.typecast(ttnn.cos(angles_cumsum32), ttnn.bfloat16)
        sin_a = ttnn.typecast(ttnn.sin(angles_cumsum32), ttnn.bfloat16)
        angles_cumsum = ttnn.typecast(angles_cumsum32, ttnn.bfloat16)

        Q_rot, Q_rot_deint = self._apply_rope(Q, cos_a, sin_a, B, T, R, H, N)
        K_rot, K_rot_deint = self._apply_rope(K, cos_a, sin_a, B, T, R, H, N)
        _t("fwd_rope", _t3)

        # --- Trapezoidal weights ---
        _t4 = _t("fwd_trap_weights")
        trap = ttnn.sigmoid(trap_raw)
        gamma = ttnn.mul(DT, trap)
        dt_shifted = self._shift_by_one(DT, B, T, H, device)
        trap_shifted = self._shift_by_one(trap, B, T, H, device)
        ones_minus_trap_shifted = ttnn.sub(self._ones_1, trap_shifted)
        shifted_gamma = ttnn.mul(dt_shifted, ones_minus_trap_shifted)
        factor = ttnn.add(gamma, shifted_gamma)

        # factor: (B,T,H) -> (B,T,1,H,1) broadcasts against (B,T,R,H,N)
        factor_5d = ttnn.reshape(factor, [B, T, 1, H, 1])
        K_scaled = ttnn.mul(K_rot, factor_5d)

        # Pre-compute backward derivatives (cached for bwd_dt_trap_qk)
        _trap_deriv = ttnn.mul(trap, ttnn.sub(self._ones_1, trap))  # sigmoid backward
        _ones_minus_trap = ttnn.sub(self._ones_1, trap)  # for shifted_gamma backward
        _t("fwd_trap_weights", _t4)

        # --- QK_dot (pre-rotation Q, K with bias) ---
        # Q_thrn and K_thrn are needed for backward; qk_dot_unscaled is not
        # computed in the reordered diagonal correction (see below).
        Q_thrn = ttnn.permute(Q, [0, 1, 3, 2, 4])  # (B,T,H,R,N)
        K_thrn = ttnn.permute(K, [0, 1, 3, 2, 4])  # (B,T,H,R,N)

        # --- MIMO up-projections ---
        _t5 = _t("fwd_mimo_proj")
        # V: (B,T,H,P) -> reshape -> (B,T,1,H,P) broadcasts against (1,1,R,H,P)
        V_5d = ttnn.reshape(V, [B, T, 1, H, P])
        z_5d = ttnn.reshape(z, [B, T, 1, H, P])
        V_proj = ttnn.mul(V_5d, self._MV_base)
        Z_proj = ttnn.mul(z_5d, self._MZ_base)
        _t("fwd_mimo_proj", _t5)

        # --- Attention (full T×T for short sequences, chunked for long) ---
        _t6 = _t("fwd_attention")
        L, L32, ADT_cumsum = self._compute_decay_matrix(ADT, B, H, T)  # kept for backward
        L_5d = ttnn.reshape(L, [B, H, 1, T, T])  # cached for backward

        Q_perm = ttnn.permute(Q_rot, [0, 3, 2, 1, 4])  # (B, H, R, T, N)
        K_perm = ttnn.permute(K_scaled, [0, 3, 2, 1, 4])
        V_perm = ttnn.permute(V_proj, [0, 3, 2, 1, 4])  # (B, H, R, T, P)

        # Use full T×T for short sequences (T*R ≤ 512 tiles), chunked for long
        # At T=128, R=4: T*R=512, which is 16×16=256 tiles — well within budget
        # At T=1024, R=4: T*R=4096, which is 128×128=16384 tiles — too large
        if T * R <= 512:
            out_attn, chunk_cache = self._full_tt_attention(
                Q_perm, K_perm, V_perm, ADT_cumsum, B, H, R, T, N, P, device)
        else:
            out_attn, chunk_cache = self._chunked_attention(
                Q_perm, K_perm, V_perm, ADT, B, H, R, T, N, P, device)
        _t("fwd_attention", _t6)

        # --- Diagonal correction (fused custom kernel) ---
        _t7 = _t("fwd_diag_corr")
        # Computes: qkv_diag = gamma * (Q @ K^T) @ V for each (B,T,H)
        # Fused into a single generic_op call — 32-45x faster than ttnn ops.
        V_proj_thrp = ttnn.permute(V_proj, [0, 1, 3, 2, 4])  # (B,T,H,R,P)

        qkv_diag = self._fused_diag_corr_forward(
            Q_thrn, K_thrn, V_proj_thrp, gamma, B, T, H, R, N, P, device)
        qkv_diag_perm = ttnn.permute(qkv_diag, [0, 2, 3, 1, 4])  # (B,H,R,T,P)
        _t("fwd_diag_corr", _t7)

        # out_ssm = strictly_causal + gamma * Q @ K^T @ V_proj
        _t8 = _t("fwd_out_proj")
        out_ssm = ttnn.add(out_attn, qkv_diag_perm)

        # --- D skip ---
        V_proj_perm = ttnn.permute(V_proj, [0, 3, 2, 1, 4])  # (B, H, R, T, P)
        # D_base (1,H,1,1,1) broadcasts against (B,H,R,T,P)
        out_with_d = ttnn.add(out_ssm, ttnn.mul(self._D_base, V_proj_perm))

        # --- Z gating ---
        Z_proj_perm = ttnn.permute(Z_proj, [0, 3, 2, 1, 4])  # (B, H, R, T, P)
        z_silu = ttnn.silu(Z_proj_perm)
        out_gated = ttnn.mul(out_with_d, z_silu)

        # --- MIMO output contraction (keep (B,H,R,T,P), no permute) ---
        # MO_base_rtp (1,H,R,1,P) broadcasts against out_gated (B,H,R,T,P)
        out_weighted = ttnn.mul(out_gated, self._MO_base_rtp)  # (B,H,R,T,P)
        out = ttnn.sum(out_weighted, dim=2)  # (B, H, T, P) — sum over R

        # --- out_proj ---
        out_flat = ttnn.reshape(ttnn.permute(out, [0, 2, 1, 3]), [B, T, self.d_inner])
        result = ttnn.linear(out_flat, self.out_proj_weight)
        _t("fwd_out_proj", _t8)

        # --- Cache ---
        self._cache = {
            "x": x, "xzBCdtAtrap": xzBCdtAtrap,
            "z": z, "V": V, "Q": Q, "K": K,
            "B_mat": B_mat, "C_mat": C_mat,
            "B_normed": B_normed, "C_normed": C_normed,
            "Q_rot": Q_rot, "K_rot": K_rot, "K_scaled": K_scaled,
            "Q_deint": Q_rot_deint, "K_deint": K_rot_deint,
            "DT": DT, "DT_exp": DT_exp, "A": A, "ADT": ADT, "trap": trap,
            "DT32": _DT32, "A32": _A32, "ones32_bth": _ones32_bth,
            "a_deriv32": _a_deriv32, "dt_deriv32": _dt_deriv32,
            "gamma": gamma, "shifted_gamma": shifted_gamma,
            "factor": factor, "factor_5d": factor_5d,
            "dt_shifted": dt_shifted, "trap_shifted": trap_shifted,
            "ones_minus_trap_shifted": ones_minus_trap_shifted,
            "trap_deriv": _trap_deriv, "ones_minus_trap": _ones_minus_trap,
            "angles_proc": angles_proc, "angles_cumsum": angles_cumsum,
            "cos_a": cos_a, "sin_a": sin_a,
            "V_proj": V_proj, "Z_proj": Z_proj, "V_5d": V_5d, "z_5d": z_5d,
            "V_proj_perm": V_proj_perm, "Z_proj_perm": Z_proj_perm,
            "V_proj_thrp": V_proj_thrp,
            "L": L, "L32": L32, "L_5d": L_5d, "out_attn": out_attn, "out_ssm": out_ssm,
            "out_with_d": out_with_d, "out_gated": out_gated, "z_silu": z_silu,
            "out_flat": out_flat,
            "Q_perm": Q_perm, "K_perm": K_perm, "V_perm": V_perm,
            "Q_thrn": Q_thrn, "K_thrn": K_thrn,
            "chunk_cache": chunk_cache,
            "qkv_diag_perm": qkv_diag_perm, "qkv_diag": qkv_diag,
            "B": B, "T": T, "H": H, "R": R, "N": N, "P": P,
        }
        return result

    # -- Backward --

    def backward(self, grad_out):
        """Manual backward. grad_out: (B, T, d_model) -> (grad_x, grads_dict)"""
        c = self._cache
        B, T, H, R = c["B"], c["T"], c["H"], c["R"]
        N, P = c["N"], c["P"]
        device = self.device
        n_angles = self.num_rope_angles
        _b0 = _t("bwd_out_proj")
        ngroups = self.ngroups

        # Constants needed in backward (scalars pre-computed in __init__)
        ones_1 = self._ones_1

        # 1. out_proj backward
        grad_out_flat = ttnn.linear(grad_out, ttnn.transpose(self.out_proj_weight, 0, 1))
        grad_out_proj_w = ttnn.matmul(
            ttnn.transpose(ttnn.reshape(c["out_flat"], [B * T, self.d_inner]), 0, 1),
            ttnn.reshape(grad_out, [B * T, self.d_model]))

        # 2. reshape: (B, T, d_inner) -> (B, H, T, P)
        grad_out_4d = ttnn.permute(ttnn.reshape(grad_out_flat, [B, T, H, P]), [0, 2, 1, 3])

        _t("bwd_out_proj", _b0)
        _b1 = _t("bwd_mimo_z_d")

        # 3. MIMO_O contraction backward (stay in (B,H,R,T,P), no permutes)
        # grad_out_exp: (B,H,1,T,P) broadcasts with MO_base_rtp (1,H,R,1,P) -> (B,H,R,T,P)
        grad_out_exp = ttnn.reshape(grad_out_4d, [B, H, 1, T, P])
        grad_out_gated = ttnn.mul(grad_out_exp, self._MO_base_rtp)  # (B,H,R,T,P)
        # dMO = sum_{b,t} grad_out * out_gated. Must use grad_out_exp, NOT
        # grad_out_gated, which already has MO folded in.
        # out_gated is cached as (B,H,R,T,P) — no permute needed
        grad_MO = ttnn.mul(grad_out_exp, c["out_gated"])  # broadcast -> (B,H,R,T,P)
        grad_MO = ttnn.sum(grad_MO, dim=(0, 3))  # (H, R, P) — sum B and T at once

        # 4. Z gating backward: out_gated = out_with_d * silu(Z_proj_perm)
        # z_silu is cached as (B,H,R,T,P) — no permute needed
        grad_out_with_d = ttnn.mul(grad_out_gated, c["z_silu"])  # (B,H,R,T,P)

        # grad_Z_proj: silu_backward(grad_out_gated * out_with_d, Z_proj_perm)
        # silu_bw fuses sigmoid(z)*(1+z-z*sigmoid(z)) into one kernel
        grad_through_silu = ttnn.mul(grad_out_gated, c["out_with_d"])
        grad_Z_proj_perm = ttnn.silu_bw(grad_through_silu, c["Z_proj_perm"])[0]

        # 5. D skip backward: out_with_d = out_ssm + D * V_proj_perm
        # grad_out_with_d is already (B, H, R, T, P) — no permute needed
        grad_out_ssm = grad_out_with_d  # (B, H, R, T, P)
        V_pp = c["V_proj_perm"]
        grad_D_full = ttnn.mul(grad_out_ssm, V_pp)
        grad_D = ttnn.sum(grad_D_full, dim=(0, 2, 3, 4))  # (H,) — sum B,R,T,P keep H
        D_bf = self._D_base  # (1,H,1,1,1) broadcasts against (B,H,R,T,P)
        grad_V_proj_from_d = ttnn.mul(grad_out_ssm, D_bf)  # (B, H, R, T, P)

        _t("bwd_mimo_z_d", _b1)
        _b2 = _t("bwd_diag_corr")

        # 6. Diagonal correction backward (original formulation):
        # Forward: qkv_diag = gamma * (Q @ K^T) @ V = gamma * QK @ V
        #   where QK = Q @ K^T (R×R), QK_g = gamma * QK (R×R)
        # Backward:
        #   grad_QK_g = grad_qkv @ V^T   -> (R,P)@(P,R) = (R,R)
        #   grad_V    = QK_g^T @ grad_qkv -> (R,R)@(R,P) = (R,P)
        #   grad_QK   = gamma * grad_QK_g -> (R,R)
        #   grad_gamma = sum(grad_QK_g * QK) -> scalar
        #   grad_Q    = grad_QK @ K         -> (R,R)@(R,N) = (R,N)
        #   grad_K    = grad_QK^T @ Q       -> (R,R)@(R,N) = (R,N)
        grad_out_attn = grad_out_ssm  # (B, H, R, T, P)
        grad_qkv_diag_perm = grad_out_ssm  # (B, H, R, T, P)
        grad_qkv_diag = ttnn.permute(grad_qkv_diag_perm, [0, 3, 1, 2, 4])  # (B, T, H, R, P)

        # Recompute QK = Q @ K^T and QK_g = gamma * QK for backward
        QK = ttnn.matmul(c["Q_thrn"], ttnn.transpose(c["K_thrn"], -2, -1),
                         compute_kernel_config=_HIFI_FP32_ACC)  # (B,T,H,R,R)
        gamma_5d = ttnn.reshape(c["gamma"], [B, T, H, 1, 1])
        QK_g = ttnn.mul(QK, gamma_5d)  # (B,T,H,R,R)

        # grad_QK_g = grad_qkv @ V^T
        grad_QK_g = ttnn.matmul(grad_qkv_diag, ttnn.transpose(c["V_proj_thrp"], -2, -1),
                                compute_kernel_config=_HIFI_FP32_ACC)  # (B,T,H,R,R)

        # grad_V from diagonal = QK_g^T @ grad_qkv
        grad_V_proj_from_diag = ttnn.matmul(ttnn.transpose(QK_g, -2, -1), grad_qkv_diag,
                                            compute_kernel_config=_HIFI_FP32_ACC)  # (B,T,H,R,P)
        grad_V_proj_from_diag = ttnn.permute(grad_V_proj_from_diag, [0, 1, 3, 2, 4])

        # grad_QK = gamma * grad_QK_g
        grad_QK = ttnn.mul(grad_QK_g, gamma_5d)  # (B,T,H,R,R)

        # grad_gamma = sum(grad_QK_g * QK) over R,R dims
        grad_gamma_from_diag = ttnn.mul(grad_QK_g, QK)
        grad_gamma_from_diag = ttnn.sum(ttnn.sum(grad_gamma_from_diag, dim=3), dim=3)  # (B,T,H)

        # grad_Q from diagonal = grad_QK @ K
        grad_Q_qkdot = ttnn.matmul(grad_QK, c["K_thrn"],
                                   compute_kernel_config=_HIFI_FP32_ACC)  # (B,T,H,R,N)

        # grad_K from diagonal = grad_QK^T @ Q
        grad_K_qkdot = ttnn.matmul(ttnn.transpose(grad_QK, -2, -1), c["Q_thrn"],
                                   compute_kernel_config=_HIFI_FP32_ACC)  # (B,T,H,R,N)

        # grad_V_proj: from D + from diag
        grad_V_proj = ttnn.add(
            ttnn.permute(grad_V_proj_from_d, [0, 3, 2, 1, 4]), grad_V_proj_from_diag)

        _t("bwd_diag_corr", _b2)
        _b3 = _t("bwd_attention")

        # 7. Attention backward (full T×T or chunked, matching forward)
        if c["T"] * c["R"] <= 512:
            grad_Q_rot, grad_K_scaled, grad_V_proj_attn, grad_L_accum, grad_ADT_from_chunk = \
                self._full_tt_attention_backward(grad_out_attn, c, B, H, R, T, N, P, device)
        else:
            grad_Q_rot, grad_K_scaled, grad_V_proj_attn, grad_L_accum, grad_ADT_from_chunk = \
                self._chunked_attention_backward(grad_out_attn, c, B, H, R, T, N, P, device)

        grad_V_proj = ttnn.add(grad_V_proj, grad_V_proj_attn)

        # Reshape grad_L_accum from 5D (B,H,1,T,T) to 4D (B,H,T,T) for decay backward
        grad_L_accum = ttnn.reshape(grad_L_accum, [B, H, T, T])

        # Combine ADT gradients from decay matrix and chunked attention
        # (grad_ADT_from_chunk will be added to grad_ADT later)

        _t("bwd_attention", _b3)
        _b4 = _t("bwd_decay_rope")

        # 8. Decay matrix backward — fp32 throughout.
        _d0 = _t("bwd_decay_matrix")
        # grad_cs_row and grad_cs_col are each sums of T terms whose difference
        # is much smaller than either, so this cancels catastrophically in bf16
        # (error grew with T: grad_A_log rel_err 0.016 @T=16 -> 0.43 @T=64).
        grad_segsum = ttnn.mul(ttnn.typecast(grad_L_accum, ttnn.float32), c["L32"])
        mask_4d = self._causal_mask(B, H, T, ttnn.float32, torch.float32)
        zeros_4d = ttnn.zeros((B, H, T, T), dtype=ttnn.float32,
                              layout=ttnn.TILE_LAYOUT, device=device)
        grad_segsum = ttnn.where(mask_4d, grad_segsum, zeros_4d)

        grad_cs_row = ttnn.sum(grad_segsum, dim=-1)
        grad_cs_col = ttnn.sum(grad_segsum, dim=-2)
        grad_cumsum = ttnn.sub(grad_cs_row, grad_cs_col)

        # reverse cumsum for ADT gradient (on-device, no host round-trip)
        grad_ADT = self._reverse_cumsum(grad_cumsum, dim=-1)
        _t("bwd_decay_matrix", _d0)

        # 9. K_scaled = K_rot * factor backward
        _d1 = _t("bwd_kscale_factor")
        # Use cached factor_5d from forward
        grad_K_rot = ttnn.mul(grad_K_scaled, c["factor_5d"])
        grad_factor = ttnn.sum(ttnn.mul(grad_K_scaled, c["K_rot"]), dim=(2, 4))  # (B, T, H) — sum R, N
        _t("bwd_kscale_factor", _d1)

        # 10. RoPE backward — grad arrives in deinterleaved layout (from forward
        # that skips interleave-back). The cached deinterleaved Q/K (Q_deint,
        # K_deint) are used directly, skipping the deinterleave permute steps.
        _d2 = _t("bwd_rope")
        def _rope_backward(grad_rotated, orig_deint, cos, sin):
            """Backward for deinterleaved RoPE.

            grad_rotated: (B,T,R,H,N) in deinterleaved layout (from attention bwd)
            orig_deint: (B,T,R,H,split) already deinterleaved (cached from forward)
            cos, sin: (B,T,H,n_ang)
            Returns: grad_orig (interleaved), grad_cos, grad_sin
            """
            split = self.split_tensor_size
            n_ang = self.num_rope_angles

            # grad is already deinterleaved — just slice the rot and pass parts
            grad_rot = ttnn.slice(grad_rotated, [0, 0, 0, 0, 0], [B, T, R, H, split])
            grad_pass = ttnn.slice(grad_rotated, [0, 0, 0, 0, split], [B, T, R, H, N])

            # orig_deint is already deinterleaved — slice directly into halves
            gr0 = ttnn.slice(grad_rot, [0, 0, 0, 0, 0], [B, T, R, H, n_ang])
            gr1 = ttnn.slice(grad_rot, [0, 0, 0, 0, n_ang], [B, T, R, H, split])
            o0 = ttnn.slice(orig_deint, [0, 0, 0, 0, 0], [B, T, R, H, n_ang])
            o1 = ttnn.slice(orig_deint, [0, 0, 0, 0, n_ang], [B, T, R, H, split])

            # Broadcast cos/sin: (B,T,H,n_ang) -> (B,T,1,H,n_ang)
            cos_e = ttnn.reshape(cos, [B, T, 1, H, n_ang])
            sin_e = ttnn.reshape(sin, [B, T, 1, H, n_ang])

            go0 = ttnn.add(ttnn.mul(gr0, cos_e), ttnn.mul(gr1, sin_e))
            go1 = ttnn.add(ttnn.neg(ttnn.mul(gr0, sin_e)), ttnn.mul(gr1, cos_e))
            gc = ttnn.add(ttnn.mul(gr0, o0), ttnn.mul(gr1, o1))
            gs = ttnn.add(ttnn.neg(ttnn.mul(gr0, o1)), ttnn.mul(gr1, o0))

            # Interleave grad back: [e0,...,e15, o0,...,o15] -> [e0,o0,e1,o1,...]
            # (needed because downstream bias backward expects interleaved layout)
            go = ttnn.concat([go0, go1], dim=-1)
            go = ttnn.reshape(go, [B, T, R, H, 2, n_ang])
            go = ttnn.permute(go, [0, 1, 2, 3, 5, 4])
            go = ttnn.reshape(go, [B, T, R, H, split])
            grad_orig = ttnn.concat([go, grad_pass], dim=-1)

            # grad_cos, grad_sin: sum over R
            grad_cos = ttnn.sum(gc, dim=2)  # (B,T,H,n_ang)
            grad_sin = ttnn.sum(gs, dim=2)
            return grad_orig, grad_cos, grad_sin

        grad_Q_pre, gc_Q, gs_Q = _rope_backward(grad_Q_rot, c["Q_deint"], c["cos_a"], c["sin_a"])
        grad_K_pre, gc_K, gs_K = _rope_backward(grad_K_rot, c["K_deint"], c["cos_a"], c["sin_a"])
        # gc/gs are (B,T,H,n_ang) — sum Q and K contributions
        grad_cos = ttnn.add(gc_Q, gc_K)
        grad_sin = ttnn.add(gs_Q, gs_K)
        _t("bwd_rope", _d2)

        # 11. cos/sin -> angles_cumsum
        _d3 = _t("bwd_angles_dt")
        grad_angles_cumsum = ttnn.add(
            ttnn.neg(ttnn.mul(c["sin_a"], grad_cos)),
            ttnn.mul(c["cos_a"], grad_sin))

        # 12. cumsum backward (reverse cumsum, on-device)
        grad_angles_scaled = self._reverse_cumsum(grad_angles_cumsum, dim=1)

        # 13. angles_scaled = angles_proc * DT
        # Use cached DT_exp from forward
        grad_angles_proc = ttnn.mul(grad_angles_scaled, c["DT_exp"])
        grad_DT_from_angles = ttnn.sum(ttnn.mul(grad_angles_scaled, c["angles_proc"]), dim=-1)

        # 14. angles_proc = tanh(angles_raw) * pi
        aop = ttnn.mul(c["angles_proc"], self._inv_pi)
        tanh_sq = ttnn.mul(aop, aop)
        ones_ap = self._ones_tensor((B, T, H, n_angles))
        grad_angles_raw = ttnn.mul(grad_angles_proc, ttnn.mul(self._pi_tt, ttnn.sub(ones_ap, tanh_sq)))
        grad_angles_raw = ttnn.sum(grad_angles_raw, dim=2)  # (B, T, n_angles)
        _t("bwd_angles_dt", _d3)
        _d4 = _t("bwd_dt_trap_qk")

        # 15. factor = gamma + shifted_gamma
        # gamma gets gradient from factor (K_scaled) and from diagonal correction
        grad_gamma = ttnn.add(grad_factor, grad_gamma_from_diag)
        # shifted_gamma only gets gradient from factor (no qkv term in new formulation)
        grad_shifted_gamma = grad_factor

        # 16. gamma = DT * trap
        grad_DT_from_gamma = ttnn.mul(grad_gamma, c["trap"])
        grad_trap_from_gamma = ttnn.mul(grad_gamma, c["DT"])

        # 17. shifted_gamma = dt_shifted * (1 - trap_shifted)
        grad_dt_shifted = ttnn.mul(grad_shifted_gamma, c["ones_minus_trap_shifted"])
        grad_trap_shifted = ttnn.neg(ttnn.mul(grad_shifted_gamma, c["dt_shifted"]))

        # 18. Shift backward
        grad_DT_from_shift = self._unshift_by_one(grad_dt_shifted, B, T, H, device)
        grad_trap_from_shift = self._unshift_by_one(grad_trap_shifted, B, T, H, device)

        # 19. Accumulate DT, trap
        grad_DT = ttnn.add(ttnn.add(grad_DT_from_angles, grad_DT_from_gamma), grad_DT_from_shift)
        grad_trap = ttnn.add(grad_trap_from_gamma, grad_trap_from_shift)

        # 20. trap = sigmoid(trap_raw) — use cached sigmoid derivative
        grad_trap_raw = ttnn.mul(grad_trap, c["trap_deriv"])

        # 21. ADT = A * DT — stay in fp32; grad_ADT arrives from the fp32
        # decay-matrix backward and feeds the B*T reductions for A_log/dt_bias.
        grad_ADT_perm = ttnn.permute(grad_ADT, [0, 2, 1])  # (B, T, H) fp32
        DT32 = c["DT32"]  # cached fp32
        A32 = c["A32"]    # cached fp32
        grad_A32 = ttnn.mul(grad_ADT_perm, DT32)
        grad_DT_from_ADT = ttnn.typecast(ttnn.mul(grad_ADT_perm, A32), ttnn.bfloat16)
        grad_DT = ttnn.add(grad_DT, grad_DT_from_ADT)

        # 22. A = -softplus(A_raw + A_log) — use cached derivative
        grad_A_pre32 = ttnn.mul(grad_A32, c["a_deriv32"])
        grad_A_raw = ttnn.typecast(grad_A_pre32, ttnn.bfloat16)  # w.r.t. A_raw
        grad_A_log = ttnn.sum(grad_A_pre32, dim=(0, 1))  # (H,) fp32

        # 23. DT = softplus(dt_raw + dt_bias) — use cached derivative
        grad_dt_pre32 = ttnn.mul(ttnn.typecast(grad_DT, ttnn.float32), c["dt_deriv32"])
        grad_dt_raw = ttnn.typecast(grad_dt_pre32, ttnn.bfloat16)
        grad_dt_bias = ttnn.sum(grad_dt_pre32, dim=(0, 1))  # (H,) fp32

        # 24. QK_dot backward — folded into section 6 (reordered diagonal correction)
        # grad_Q_qkdot and grad_K_qkdot are already computed in section 6.
        # Permute from (B,T,H,R,N) to (B,T,R,H,N)
        grad_Q_qkdot = ttnn.permute(grad_Q_qkdot, [0, 1, 3, 2, 4])
        grad_K_qkdot = ttnn.permute(grad_K_qkdot, [0, 1, 3, 2, 4])

        grad_Q_pre = ttnn.add(grad_Q_pre, grad_Q_qkdot)
        grad_K_pre = ttnn.add(grad_K_pre, grad_K_qkdot)

        # 25. Q = C_normed + C_bias, K = B_normed + B_bias
        # grad_Q_pre: (B, T, R, H, N) -> sum over B, T -> (R, H, N) -> permute -> (H, R, N)
        grad_C_bias = ttnn.permute(
            ttnn.sum(grad_Q_pre, dim=(0, 1)), [1, 0, 2])
        grad_C_normed = grad_Q_pre
        grad_B_bias = ttnn.permute(
            ttnn.sum(grad_K_pre, dim=(0, 1)), [1, 0, 2])
        grad_B_normed = grad_K_pre

        # 26. QKNorm backward — pass 5D directly (TTRMSNorm.backward handles arbitrary rank)
        _t("bwd_dt_trap_qk", _d4)
        _d5 = _t("bwd_qk_norm")
        # TTRMSNorm.backward(grad_out, x) expects x to be the PRE-norm input —
        # it recomputes rms and x_normed from it. Passing the post-norm output
        # gives rms ~= 1 and silently wrong grads.
        grad_B_raw, grad_B_norm_w = self.B_norm.backward(grad_B_normed, c["B_mat"])
        grad_C_raw, grad_C_norm_w = self.C_norm.backward(grad_C_normed, c["C_mat"])

        _t("bwd_qk_norm", _d5)
        _t("bwd_decay_rope", _b4)
        _b5 = _t("bwd_in_proj")

        # 27. MIMO up-projection backward (element-wise: V_proj = V * MIMO_V)
        # MV_base (1,1,R,H,P) broadcasts against grad_V_proj (B,T,R,H,P)
        grad_V_exp = ttnn.mul(grad_V_proj, self._MV_base)  # (B, T, R, H, P)
        grad_V = ttnn.sum(grad_V_exp, dim=2)  # (B, T, H, P) — sum over R

        # Use cached V_5d from forward
        grad_MV_full = ttnn.mul(grad_V_proj, c["V_5d"])  # (B, T, R, H, P)
        grad_MIMO_V = ttnn.permute(ttnn.sum(grad_MV_full, dim=(0, 1)), [1, 0, 2])

        # Z_proj backward (same structure)
        # grad_Z_proj_perm is (B, H, R, T, P), need (B, T, R, H, P)
        grad_Z_proj_trh = ttnn.permute(grad_Z_proj_perm, [0, 3, 2, 1, 4])  # (B, T, R, H, P)
        grad_z_exp = ttnn.mul(grad_Z_proj_trh, self._MZ_base)  # broadcast
        grad_z = ttnn.sum(grad_z_exp, dim=2)  # (B, T, H, P)

        # Use cached z_5d from forward
        grad_MZ_full = ttnn.mul(grad_Z_proj_trh, c["z_5d"])  # broadcast
        grad_MIMO_Z = ttnn.permute(ttnn.sum(grad_MZ_full, dim=(0, 1)), [1, 0, 2])

        # 28. Reshape backward: V, z (B, T, H, P) -> x_raw, z_raw
        grad_x_raw = ttnn.reshape(grad_V, [B, T, self.d_inner])
        grad_z_raw = ttnn.reshape(grad_z, [B, T, self.d_inner])

        # B_raw, C_raw: GQA sum then reshape
        if ngroups < H:
            grad_B_raw = ttnn.sum(
                ttnn.reshape(grad_B_raw, [B, T, R, ngroups, H // ngroups, N]), dim=4)
            grad_C_raw = ttnn.sum(
                ttnn.reshape(grad_C_raw, [B, T, R, ngroups, H // ngroups, N]), dim=4)
        grad_B_raw = ttnn.reshape(grad_B_raw, [B, T, self.d_state * ngroups * R])
        grad_C_raw = ttnn.reshape(grad_C_raw, [B, T, self.d_state * ngroups * R])

        # 29. Concat split gradients
        grad_xzBCdtAtrap = ttnn.concat([
            grad_z_raw, grad_x_raw, grad_B_raw, grad_C_raw,
            grad_dt_raw, grad_A_raw, grad_trap_raw, grad_angles_raw
        ], dim=-1)

        # 30. in_proj backward
        grad_x = ttnn.linear(grad_xzBCdtAtrap, ttnn.transpose(self.in_proj_weight, 0, 1))
        grad_in_proj_w = ttnn.matmul(
            ttnn.transpose(ttnn.reshape(c["x"], [B * T, self.d_model]), 0, 1),
            ttnn.reshape(grad_xzBCdtAtrap, [B * T, grad_xzBCdtAtrap.shape[-1]]))

        # Collect gradients
        grads = {
            "in_proj_weight": grad_in_proj_w,
            "out_proj_weight": grad_out_proj_w,
            "dt_bias": ttnn.typecast(grad_dt_bias, ttnn.float32),
            "A_log": ttnn.typecast(grad_A_log, ttnn.float32),
            "D": ttnn.typecast(grad_D, ttnn.float32),
            "B_bias": ttnn.typecast(grad_B_bias, ttnn.float32),
            "C_bias": ttnn.typecast(grad_C_bias, ttnn.float32),
            "B_norm_weight": grad_B_norm_w,
            "C_norm_weight": grad_C_norm_w,
            "MIMO_V": ttnn.typecast(grad_MIMO_V, ttnn.float32),
            "MIMO_Z": ttnn.typecast(grad_MIMO_Z, ttnn.float32),
            "MIMO_O": ttnn.typecast(grad_MO, ttnn.float32),
        }
        _t("bwd_in_proj", _b5)
        return grad_x, grads

    # -- Parameter management --

    def get_params(self):
        return {
            "in_proj_weight": self.in_proj_weight,
            "out_proj_weight": self.out_proj_weight,
            "dt_bias": self.dt_bias,
            "A_log": self.A_log,
            "D": self.D,
            "B_bias": self.B_bias,
            "C_bias": self.C_bias,
            "B_norm_weight": self.B_norm.weight,
            "C_norm_weight": self.C_norm.weight,
            "MIMO_V": self.MIMO_V,
            "MIMO_Z": self.MIMO_Z,
            "MIMO_O": self.MIMO_O,
        }

    def set_params(self, params):
        for k, v in params.items():
            if k == "B_norm_weight":
                self.B_norm.weight = v
            elif k == "C_norm_weight":
                self.C_norm.weight = v
            elif hasattr(self, k):
                setattr(self, k, v)
        # Re-compute pre-computed parameter bases after parameter updates
        self._recompute_bases()

    def _recompute_bases(self):
        """Recompute pre-computed parameter base tensors from current params."""
        H, R, P, N = self.nheads, self.R, self.headdim, self.d_state
        self._MV_base = ttnn.typecast(
            ttnn.reshape(ttnn.permute(self.MIMO_V, [1, 0, 2]), [1, 1, R, H, P]),
            ttnn.bfloat16)
        self._MZ_base = ttnn.typecast(
            ttnn.reshape(ttnn.permute(self.MIMO_Z, [1, 0, 2]), [1, 1, R, H, P]),
            ttnn.bfloat16)
        self._MO_base = ttnn.typecast(
            ttnn.reshape(self.MIMO_O, [1, H, 1, R, P]),
            ttnn.bfloat16)
        self._MO_base_rtp = ttnn.typecast(
            ttnn.reshape(self.MIMO_O, [1, H, R, 1, P]),
            ttnn.bfloat16)
        self._B_bias_base = ttnn.typecast(
            ttnn.reshape(ttnn.permute(self.B_bias, [1, 0, 2]), [1, 1, R, H, N]),
            ttnn.bfloat16)
        self._C_bias_base = ttnn.typecast(
            ttnn.reshape(ttnn.permute(self.C_bias, [1, 0, 2]), [1, 1, R, H, N]),
            ttnn.bfloat16)
        self._D_base = ttnn.typecast(
            ttnn.reshape(self.D, [1, H, 1, 1, 1]),
            ttnn.bfloat16)


def _print_m3_timings(n_layers):
    """Print per-section timing summary. Call after a training step."""
    if not _M3_PROFILE or not _M3_TIMINGS:
        return
    total = sum(_M3_TIMINGS.values())
    print(f"\n--- Mamba-3 layer timing (per-layer avg, {n_layers} layers/step) ---")
    for name, t in sorted(_M3_TIMINGS.items(), key=lambda x: -x[1]):
        avg = t / n_layers
        pct = 100.0 * t / total if total > 0 else 0
        print(f"  {name:25s}: {avg*1000:8.2f} ms/layer  ({pct:5.1f}%)")
    print(f"  {'TOTAL':25s}: {total*1000/n_layers:8.2f} ms/layer")
    print(f"  Per-step (x{n_layers}): {total:8.3f} s\n")
