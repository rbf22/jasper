// SPDX-License-Identifier: Apache-2.0
// Fused diagonal correction writer: writes output tiles to DRAM
//
// Runtime args:
//   0: dst_addr
//   1: num_batch (batch elements for this core)
//   2: batch_start_id (global batch index for this core's first element)
//   3: batch_stride (output tiles per batch, = Pt = 2)
//
// Compile-time args (via TensorAccessorArgs):
//   [0]: out_args...

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_batch = get_arg_val<uint32_t>(1);
    uint32_t batch_start_id = get_arg_val<uint32_t>(2);
    uint32_t batch_stride = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr auto out_args = TensorAccessorArgs<0>();

    const uint32_t out_tile_bytes = get_tile_size(cb_out);
    const auto s_out = TensorAccessor(out_args, dst_addr);

    Noc noc;
    CircularBuffer out_cb(cb_out);

    constexpr uint32_t onetile = 1;

    for (uint32_t b = 0; b < num_batch; b++) {
        uint32_t batch_id = batch_start_id + b;
        uint32_t out_tile_base = batch_id * batch_stride;

        for (uint32_t t = 0; t < batch_stride; t++) {
            out_cb.wait_front(onetile);
            noc.async_write(out_cb, s_out, out_tile_bytes, {}, {.page_id = out_tile_base + t});
            noc.async_writes_flushed();
            out_cb.pop_front(onetile);
        }
    }
    noc.async_write_barrier();
}
