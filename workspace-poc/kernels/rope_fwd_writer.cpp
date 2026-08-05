// SPDX-License-Identifier: Apache-2.0
// Fused RoPE forward writer: writes rot_0 and rot_1 tiles to DRAM
//
// Runtime args:
//   0: dst_rot0_addr
//   1: dst_rot1_addr
//   2: num_batch (batch elements for this core)
//   3: batch_start_id (global batch index for this core's first element)
//   4: batch_stride (output tiles per batch, = Nt)
//
// Compile-time args (via TensorAccessorArgs):
//   [0]: rot0_args...
//   [rot0_offset]: rot1_args...

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_rot0_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_rot1_addr = get_arg_val<uint32_t>(1);
    uint32_t num_batch = get_arg_val<uint32_t>(2);
    uint32_t batch_start_id = get_arg_val<uint32_t>(3);
    uint32_t batch_stride = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_rot0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_rot1 = tt::CBIndex::c_17;

    constexpr auto rot0_args = TensorAccessorArgs<0>();
    constexpr auto rot1_args = TensorAccessorArgs<rot0_args.next_compile_time_args_offset()>();

    const uint32_t rot0_tile_bytes = get_tile_size(cb_rot0);
    const uint32_t rot1_tile_bytes = get_tile_size(cb_rot1);

    const auto s_rot0 = TensorAccessor(rot0_args, dst_rot0_addr);
    const auto s_rot1 = TensorAccessor(rot1_args, dst_rot1_addr);

    Noc noc;
    CircularBuffer rot0_cb(cb_rot0);
    CircularBuffer rot1_cb(cb_rot1);

    constexpr uint32_t onetile = 1;

    for (uint32_t b = 0; b < num_batch; b++) {
        uint32_t batch_id = batch_start_id + b;
        uint32_t tile_id = batch_id * batch_stride;

        // Write rot0
        rot0_cb.wait_front(onetile);
        noc.async_write(rot0_cb, s_rot0, rot0_tile_bytes, {}, {.page_id = tile_id});
        noc.async_writes_flushed();
        rot0_cb.pop_front(onetile);

        // Write rot1
        rot1_cb.wait_front(onetile);
        noc.async_write(rot1_cb, s_rot1, rot1_tile_bytes, {}, {.page_id = tile_id});
        noc.async_writes_flushed();
        rot1_cb.pop_front(onetile);
    }
    noc.async_write_barrier();
}
