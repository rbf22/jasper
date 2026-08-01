// SPDX-License-Identifier: Apache-2.0
// Fused RoPE forward reader: reads t0, t1, cos, sin tiles from DRAM
//
// t0, t1 have shape (B,T,R,H,n_angles) → B*T*R tiles (H and n_angles packed in one 32x32 tile)
// cos, sin have shape (B,T,H,n_angles) → B*T tiles (broadcast over R)
//
// For batch element i (iterating over B*T*R):
//   t0/t1 tile_id = i
//   cos/sin tile_id = i / R  (integer division — broadcasts over R)
//
// Runtime args:
//   0: src_t0_addr
//   1: src_t1_addr
//   2: src_cos_addr
//   3: src_sin_addr
//   4: num_batch (B*T*R elements for this core)
//   5: batch_start_id (global tile index for this core's first element)
//   6: R (for cos/sin tile index computation)
//
// Compile-time args (via TensorAccessorArgs):
//   [0]: t0_args...
//   [t0_offset]: t1_args...
//   [t1_offset]: cos_args...
//   [cos_offset]: sin_args...

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src_t0_addr = get_arg_val<uint32_t>(0);
    uint32_t src_t1_addr = get_arg_val<uint32_t>(1);
    uint32_t src_cos_addr = get_arg_val<uint32_t>(2);
    uint32_t src_sin_addr = get_arg_val<uint32_t>(3);
    uint32_t num_batch = get_arg_val<uint32_t>(4);
    uint32_t batch_start_id = get_arg_val<uint32_t>(5);
    uint32_t R = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_t0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_t1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_cos = tt::CBIndex::c_2;
    constexpr uint32_t cb_sin = tt::CBIndex::c_3;

    constexpr auto t0_args = TensorAccessorArgs<0>();
    constexpr auto t1_args = TensorAccessorArgs<t0_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<t1_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    const uint32_t t0_tile_bytes = get_tile_size(cb_t0);
    const uint32_t t1_tile_bytes = get_tile_size(cb_t1);
    const uint32_t cos_tile_bytes = get_tile_size(cb_cos);
    const uint32_t sin_tile_bytes = get_tile_size(cb_sin);

    const auto s_t0 = TensorAccessor(t0_args, src_t0_addr);
    const auto s_t1 = TensorAccessor(t1_args, src_t1_addr);
    const auto s_cos = TensorAccessor(cos_args, src_cos_addr);
    const auto s_sin = TensorAccessor(sin_args, src_sin_addr);

    Noc noc;
    CircularBuffer cbt0(cb_t0);
    CircularBuffer cbt1(cb_t1);
    CircularBuffer cbcos(cb_cos);
    CircularBuffer cbsin(cb_sin);

    constexpr uint32_t onetile = 1;

    for (uint32_t b = 0; b < num_batch; b++) {
        uint32_t batch_id = batch_start_id + b;
        uint32_t t0_tile_id = batch_id;
        uint32_t cos_tile_id = batch_id / R;

        // Push t0 twice (for t0*cos and t0*sin)
        for (uint32_t rep = 0; rep < 2; rep++) {
            cbt0.reserve_back(onetile);
            noc.async_read(s_t0, cbt0, t0_tile_bytes, {.page_id = t0_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cbt0.push_back(onetile);
        }

        // Push t1 twice (for t1*sin and t1*cos)
        for (uint32_t rep = 0; rep < 2; rep++) {
            cbt1.reserve_back(onetile);
            noc.async_read(s_t1, cbt1, t1_tile_bytes, {.page_id = t0_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cbt1.push_back(onetile);
        }

        // Push cos twice (for t0*cos and t1*cos)
        for (uint32_t rep = 0; rep < 2; rep++) {
            cbcos.reserve_back(onetile);
            noc.async_read(s_cos, cbcos, cos_tile_bytes, {.page_id = cos_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cbcos.push_back(onetile);
        }

        // Push sin twice (for t1*sin and t0*sin)
        for (uint32_t rep = 0; rep < 2; rep++) {
            cbsin.reserve_back(onetile);
            noc.async_read(s_sin, cbsin, sin_tile_bytes, {.page_id = cos_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cbsin.push_back(onetile);
        }
    }
}
