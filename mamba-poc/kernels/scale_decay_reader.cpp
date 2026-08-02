// SPDX-License-Identifier: Apache-2.0
// Fused scale + decay kernel: scores = scores_raw * scale * D_decay
//
// Reader: reads scores_raw tiles and D_decay tiles (broadcast over batch)
// Compute: mul_tiles(scores_raw, D) then mul_unary(scale)
// Writer: writes output tiles to DRAM
//
// Runtime args:
//   0: scores_raw_addr
//   1: D_decay_addr
//   2: num_tiles (tiles for this core)
//   3: tile_start_id (global tile index for this core's first tile)
//   4: D_tiles_per_row (T / 32, number of tile columns in D_decay)
//   5: HT_tiles (H*T / 32, number of tile rows in D_decay — for broadcast)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t scores_raw_addr = get_arg_val<uint32_t>(0);
    uint32_t D_decay_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t tile_start_id = get_arg_val<uint32_t>(3);
    uint32_t D_tiles_per_row = get_arg_val<uint32_t>(4);
    uint32_t HT_tiles = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_scores = tt::CBIndex::c_0;
    constexpr uint32_t cb_D = tt::CBIndex::c_1;

    constexpr auto scores_args = TensorAccessorArgs<0>();
    constexpr auto D_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();

    const uint32_t scores_tile_bytes = get_tile_size(cb_scores);
    const uint32_t D_tile_bytes = get_tile_size(cb_D);

    const auto s_scores = TensorAccessor(scores_args, scores_raw_addr);
    const auto s_D = TensorAccessor(D_args, D_decay_addr);

    Noc noc;
    CircularBuffer scores_cb(cb_scores);
    CircularBuffer D_cb(cb_D);

    constexpr uint32_t onetile = 1;

    for (uint32_t i = 0; i < num_tiles; i++) {
        uint32_t tile_id = tile_start_id + i;
        uint32_t row = tile_id / D_tiles_per_row;
        uint32_t col = tile_id % D_tiles_per_row;
        // D_decay is (H*T, T) in 2D. Broadcast over B: D tile row = row % HT_tiles
        uint32_t D_tile_id = (row % HT_tiles) * D_tiles_per_row + col;

        // Read scores_raw tile
        scores_cb.reserve_back(onetile);
        noc.async_read(s_scores, scores_cb, scores_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        scores_cb.push_back(onetile);

        // Read D_decay tile (broadcast)
        D_cb.reserve_back(onetile);
        noc.async_read(s_D, D_cb, D_tile_bytes, {.page_id = D_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        D_cb.push_back(onetile);
    }
}
