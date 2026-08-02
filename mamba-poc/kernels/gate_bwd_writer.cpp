// SPDX-License-Identifier: Apache-2.0
// Fused gate backward writer — writes 2 output tiles per iteration.
//
// Runtime args:
//   0: dst_out0_addr
//   1: dst_out1_addr
//   2: num_tiles (tiles for this core)
//   3: tile_start_id
//
// Compile-time args (via TensorAccessorArgs):
//   [0]: out0_args...
//   [out0_offset]: out1_args...

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_out0_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_out1_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t tile_start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_out1 = tt::CBIndex::c_17;

    constexpr auto out0_args = TensorAccessorArgs<0>();
    constexpr auto out1_args = TensorAccessorArgs<out0_args.next_compile_time_args_offset()>();

    const uint32_t out0_tile_bytes = get_tile_size(cb_out0);
    const uint32_t out1_tile_bytes = get_tile_size(cb_out1);

    const auto s_out0 = TensorAccessor(out0_args, dst_out0_addr);
    const auto s_out1 = TensorAccessor(out1_args, dst_out1_addr);

    Noc noc;
    CircularBuffer out0_cb(cb_out0);
    CircularBuffer out1_cb(cb_out1);

    constexpr uint32_t onetile = 1;

    for (uint32_t i = 0; i < num_tiles; i++) {
        uint32_t tile_id = tile_start_id + i;

        out0_cb.wait_front(onetile);
        noc.async_write(out0_cb, s_out0, out0_tile_bytes, {}, {.page_id = tile_id});
        noc.async_write_barrier();
        out0_cb.pop_front(onetile);

        out1_cb.wait_front(onetile);
        noc.async_write(out1_cb, s_out1, out1_tile_bytes, {}, {.page_id = tile_id});
        noc.async_write_barrier();
        out1_cb.pop_front(onetile);
    }
}
