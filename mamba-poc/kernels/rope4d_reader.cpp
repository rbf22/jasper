// SPDX-License-Identifier: Apache-2.0
// Fused RoPE reader for 4D retention tensors.
//
// t0, t1 have shape (B, H, T, d_h//2) → tiled as (B*H*T, tiles_per_row)
// cos, sin have shape (T, d_h//2) → tiled as (T_tiles, tiles_per_row)
//
// cos/sin broadcast over B and H. For x tile at global tile_id:
//   row = tile_id / tiles_per_row
//   col = tile_id % tiles_per_row
//   cos_tile_id = (row % T_tiles) * tiles_per_row + col
//
// Runtime args:
//   0: src_t0_addr
//   1: src_t1_addr
//   2: src_cos_addr
//   3: src_sin_addr
//   4: num_tiles (tiles for this core)
//   5: tile_start_id (global tile index for this core's first tile)
//   6: T_tiles (T / 32, number of tile rows in cos/sin)
//   7: tiles_per_row (tiles per row in t0/t1/cos/sin)
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
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    uint32_t tile_start_id = get_arg_val<uint32_t>(5);
    uint32_t T_tiles = get_arg_val<uint32_t>(6);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(7);

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

    for (uint32_t i = 0; i < num_tiles; i++) {
        uint32_t tile_id = tile_start_id + i;
        uint32_t row = tile_id / tiles_per_row;
        uint32_t col = tile_id % tiles_per_row;
        // cos/sin are (T, d_half) in TILE_LAYOUT. T must be a multiple of 32
        // so each x tile row maps to exactly one cos tile row.
        // cos_tile_row = row % T_tiles (broadcast over B, H)
        uint32_t cos_tile_id = (row % T_tiles) * tiles_per_row + col;

        // Push t0, t1, cos, sin — one each (simplified kernel)
        cbt0.reserve_back(onetile);
        noc.async_read(s_t0, cbt0, t0_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbt0.push_back(onetile);

        cbt1.reserve_back(onetile);
        noc.async_read(s_t1, cbt1, t1_tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbt1.push_back(onetile);

        cbcos.reserve_back(onetile);
        noc.async_read(s_cos, cbcos, cos_tile_bytes, {.page_id = cos_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbcos.push_back(onetile);

        cbsin.reserve_back(onetile);
        noc.async_read(s_sin, cbsin, sin_tile_bytes, {.page_id = cos_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cbsin.push_back(onetile);
    }
}
