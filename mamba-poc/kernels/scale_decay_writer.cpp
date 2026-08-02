// SPDX-License-Identifier: Apache-2.0
// Fused scale + decay writer: writes output tiles to DRAM
//
// Runtime args:
//   0: dst_addr
//   1: num_tiles (tiles for this core)
//   2: tile_start_id (global tile index for this core's first tile)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t tile_start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr auto out_args = TensorAccessorArgs<0>();

    const uint32_t out_tile_bytes = get_tile_size(cb_out);
    const auto s_out = TensorAccessor(out_args, dst_addr);

    Noc noc;
    CircularBuffer out_cb(cb_out);

    constexpr uint32_t onetile = 1;

    for (uint32_t i = 0; i < num_tiles; i++) {
        uint32_t tile_id = tile_start_id + i;
        out_cb.wait_front(onetile);
        noc.async_write(out_cb, s_out, out_tile_bytes, {}, {.page_id = tile_id});
        noc.async_write_barrier();
        out_cb.pop_front(onetile);
    }
}
