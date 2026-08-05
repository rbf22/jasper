// SPDX-License-Identifier: Apache-2.0
// Fused diagonal correction reader: reads Q, K, V, gamma tiles from DRAM
//
// For each (B,T,H) batch element, reads:
//   Q: Nt tiles (R padded to 32, N=64 → 1×2 tiles)
//   K: Nt tiles (same as Q)
//   V: Pt tiles (R padded to 32, P=64 → 1×2 tiles)
//   gamma: 1 tile (scalar padded to 32×32)
//
// Runtime args:
//   0: src_q_addr
//   1: src_k_addr
//   2: src_v_addr
//   3: src_gamma_addr
//   4: num_batch_elements (for this core)
//   5: batch_start_id (global batch index for this core's first element)
//   6: batch_stride (tiles per batch element in Q/K/V, = Nt = Pt = 2)
//   7: gamma_batch_stride (tiles per batch in gamma, = 1)
//
// Compile-time args (via TensorAccessorArgs):
//   [0]: q_args... (TensorAccessorArgs for Q)
//   [q_offset]: k_args...
//   [k_offset]: v_args...
//   [v_offset]: gamma_args...

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src_q_addr = get_arg_val<uint32_t>(0);
    uint32_t src_k_addr = get_arg_val<uint32_t>(1);
    uint32_t src_v_addr = get_arg_val<uint32_t>(2);
    uint32_t src_gamma_addr = get_arg_val<uint32_t>(3);
    uint32_t num_batch = get_arg_val<uint32_t>(4);
    uint32_t batch_start_id = get_arg_val<uint32_t>(5);
    uint32_t batch_stride = get_arg_val<uint32_t>(6);  // tiles per batch in Q/K/V
    uint32_t gamma_batch_stride = get_arg_val<uint32_t>(7);  // tiles per batch in gamma

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_3;

    constexpr auto q_args = TensorAccessorArgs<0>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    const uint32_t q_tile_bytes = get_tile_size(cb_q);
    const uint32_t k_tile_bytes = get_tile_size(cb_k);
    const uint32_t v_tile_bytes = get_tile_size(cb_v);
    const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);

    const auto s_q = TensorAccessor(q_args, src_q_addr);
    const auto s_k = TensorAccessor(k_args, src_k_addr);
    const auto s_v = TensorAccessor(v_args, src_v_addr);
    const auto s_gamma = TensorAccessor(gamma_args, src_gamma_addr);

    Noc noc;
    CircularBuffer cbq(cb_q);
    CircularBuffer cbk(cb_k);
    CircularBuffer cbv(cb_v);
    CircularBuffer cbg(cb_gamma);

    constexpr uint32_t onetile = 1;

    for (uint32_t b = 0; b < num_batch; b++) {
        uint32_t batch_id = batch_start_id + b;
        uint32_t qk_tile_base = batch_id * batch_stride;
        uint32_t gamma_tile_id = batch_id * gamma_batch_stride;

        // Read Q tiles (batch_stride tiles)
        for (uint32_t t = 0; t < batch_stride; t++) {
            cbq.reserve_back(onetile);
            noc.async_read(s_q, cbq, q_tile_bytes, {.page_id = qk_tile_base + t}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cbq.push_back(onetile);
        }

        // Read K tiles (batch_stride tiles)
        for (uint32_t t = 0; t < batch_stride; t++) {
            cbk.reserve_back(onetile);
            noc.async_read(s_k, cbk, k_tile_bytes, {.page_id = qk_tile_base + t}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cbk.push_back(onetile);
        }

        // Read V tiles (batch_stride tiles, same as Q/K since Nt=Pt=2)
        for (uint32_t t = 0; t < batch_stride; t++) {
            cbv.reserve_back(onetile);
            noc.async_read(s_v, cbv, v_tile_bytes, {.page_id = qk_tile_base + t}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cbv.push_back(onetile);
        }

        // Read gamma tile (1 tile)
        cbg.reserve_back(onetile);
        noc.async_read(s_gamma, cbg, gamma_tile_bytes, {.page_id = gamma_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbg.push_back(onetile);
    }
}
