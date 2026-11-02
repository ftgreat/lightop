// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
//
// moe_w8a8_group128.cu — INT8 W8A8 group-128 MoE GEMM
// (全新文件: 不修改 lightop 任何既有源码 / 绑定 / 编译清单)
//
// 支持压缩权重格式 (compressed-tensors int-quantized group-128 原始布局):
//   qweight [E, N, K] int8 (未重排, contiguous)
//   b_scale [E, N, K/128]  fp32
//   a_scale [*,  K/128]    fp32   (per-token-group)
//
// 计算语义 (与 lightop Triton fused_moe_kernel g128 分支一致):
//   out[m,n] = Σ_g ( Σ_{k∈g} a_q[m,k]·b_q[e,n,k] ) · a_s[m,g] · b_s[e,n,g]
//   DOWN 阶段乘 topk_weights[pair]
//
// kernel 结构:
//   * grid: (1, N/BLOCK_N, sorted/BLOCK_M), 64-lane warp
//   * 每 warp 处理 [16 x 128K] A tile × [16 x 128K] B tile
//     (WARP_M=WARP_N=16, WARP_K=128 = group_size, warp_k_num=1)
//   * lane = row_id(16) × col_id(4); 每 lane 16B int8 = 16 个 K
//   * 128K = 2 次 load (各 64K) + 4 次 mmac_i32_16x16x32 (各 32K)
//   * 组级双缓冲: 计算组 g 时预取组 g+1 (延迟隐藏, 组边界严格对齐)
//   * int32 组部分和 × a_scale[m,g] × b_scale[e,n,g] 累加进 fp32
//   * DOWN 阶段 topk_weight 乘在折算处 (等价于乘最终和, 数值一致)
//   * 写回: 共享内存中转 + dwordx4 scatter (与既有 DECODE kernel 一致)
//
// 入口签名与 moe_gemm_w8a8 保持一致; topk_weights 有值 → DOWN(gemm2)。

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <cstdint>

#include "moe_w8a8_utils_hip.h"

#ifndef DIVIDE
#define DIVIDE(a, b) (((a) + (b) - 1) / (b))
#endif

namespace at::native {
namespace moe_g128 {
// HIP int4 → 普通POD int4 (兼容 vec<int,4>::type 赋值)
__device__ __forceinline__ intx4 make_int4_like(const int4& v) {
    intx4 r;
    int tmp[4] = {v.x, v.y, v.z, v.w};
    r = *(intx4*)tmp;
    return r;
}


__device__ __forceinline__ int2 g128_half(int4 v, int half) {
    int2 r;
    r.x = half == 0 ? v.x : v.z;
    r.y = half == 0 ? v.y : v.w;
    return r;
}

constexpr int G128 = 128;  // group size

// 全局 debug 缓冲 (host 可拷出): [block_id, threadIdx, row_a, row_b,
//   a_scale_ptr_off, b_scale_ptr_off, n_groups, sid, token, first_load_val]
__device__ int g128_dbg_buf[16 * 1024 * 10];
__global__ void g128_dbg_clear() { for (int i = threadIdx.x; i < 16*1024*10; i += blockDim.x) g128_dbg_buf[i] = -7; }
int* g128_dbg_ptr() { return nullptr; }

// ---------------------------------------------------------------------------
// 单组核心: [16 x 128K] × [128K x 16] → int32 部分和 → 折算进 acc
// a_reg/b_reg: 双缓冲 (2 x 16B/lane); 一次 load 64K → 2 loads 覆盖 128K
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 少组直连版: n_groups 小 (<8) 时屏障开销 > A 复用收益, 直接全局 load.
// 与 LDS 版数学等价 (同一 16B load + 8B chunk mmac 语义).
// ---------------------------------------------------------------------------
template<typename Element, bool MUL_TOPK, int MT>
__forceinline__ __device__ void g128_tile_direct(
    const Element* __restrict__ A_ptr,
    const Element* __restrict__ B_ptr,
    const float**  a_scale_rows,
    const float*   b_scale_base,
    int            b_scale_noff,
    int            bsn,
    int            n_groups,
    int*           row_a_arr,
    int            row_b,
    float*         topk_w_arr,
    floatx4*       acc,
    int            lane_col)
{
    int4 na[MT][2], nb[2];
    #pragma unroll
    for(int mt = 0; mt < MT; ++mt){
        const Element* pa = A_ptr + row_a_arr[mt];
        na[mt][0] = *(const int4*)(pa);
        na[mt][1] = *(const int4*)(pa + 64);
    }
    {
        const Element* pb = B_ptr + row_b;
        nb[0] = *(const int4*)(pb);
        nb[1] = *(const int4*)(pb + 64);
    }
    for(int g = 0; g < n_groups; ++g){
        intx4 c[MT];
        #pragma unroll
        for(int mt = 0; mt < MT; ++mt) c[mt] = intx4{0,0,0,0};
        int4 ta[MT][2], tb[2];
        const int kn = (g + 1 < n_groups ? g + 1 : n_groups - 1) * 128;
        #pragma unroll
        for(int mt = 0; mt < MT; ++mt){
            const Element* pa = A_ptr + row_a_arr[mt] + kn;
            ta[mt][0] = *(const int4*)(pa);
            ta[mt][1] = *(const int4*)(pa + 64);
        }
        {
            const Element* pb = B_ptr + row_b + kn;
            tb[0] = *(const int4*)(pb);
            tb[1] = *(const int4*)(pb + 64);
        }
        #pragma unroll
        for(int mt = 0; mt < MT; ++mt){
            const int* ai = (const int*)&na[mt][0];
            const int* bi = (const int*)&nb[0];
            #pragma unroll
            for(int j = 0; j < 4; ++j){
                int2 va = make_int2(ai[2*j], ai[2*j+1]);
                int2 vb = make_int2(bi[2*j], bi[2*j+1]);
                c[mt] = mmac<Element>(
                    *(typename vec<Element,8>::type*)&va,
                    *(typename vec<Element,8>::type*)&vb,
                    c[mt]);
            }
        }
        #pragma unroll
        for(int mt = 0; mt < MT; ++mt){
            float a_s = *(a_scale_rows[mt] + g);
            float w = MUL_TOPK ? topk_w_arr[mt] : 1.0f;
            #pragma unroll
            for(int i = 0; i < 4; ++i) {
                float b_s = *(b_scale_base + (b_scale_noff + lane_col + i * 4) * bsn + g);
                acc[mt][i] += (float)c[mt][i] * a_s * b_s * w;
            }
        }
        #pragma unroll
        for(int mt = 0; mt < MT; ++mt){
            na[mt][0] = ta[mt][0]; na[mt][1] = ta[mt][1];
        }
        nb[0] = tb[0]; nb[1] = tb[1];
    }
}

template<typename Element, bool MUL_TOPK, int MT>
__forceinline__ __device__ void g128_tile(
    const Element* __restrict__ A_ptr,
    const Element* __restrict__ B_ptr,
    const float**  a_scale_rows,   // [MT]
    const float*   b_scale_base,
    int            b_scale_noff,   // n_base = bidy*BLOCK_N + warp_n*16
    int            bsn,
    int            n_groups,
    int*           row_a_arr,      // [MT] token*size_k + col_id*16
    int            row_b,          // n_row*size_k + col_id*16
    float*         topk_w_arr,     // [MT]
    floatx4*       acc,            // [MT]
    int            lane_row,
    int            lane_col,
    char*          a_smem)
{
    // MT 个 m-tile (每 16 token), 共享同一 B 数据 — B 流量 ÷ MT.
    // 16B 连续 load + 8B chunk mmac (与已验证的 8B 模式组部分和逐位等价):
    //   lane c 每组 load 2×16B: k [col*16,+16) 与 [col*16+64,+16) → 32B
    //   mmac j (j=lt*2+kh) 用 chunk j (8B): k 集 {col*16 + 64lt + 8kh} —
    //   j=0..3 并集 = [0,128) 全覆盖, A/B 同 chunk 配对 → 组 int32 和一致.
    // A 经 LDS 复用: 4 warp 的 lane_id 相同 → row_a/token 相同 →
    // warp0 load A 全局数据写 LDS, 所有 warp 从 LDS 读 (A 流量 ÷4).
    // B 各 warp 独立 (不同 N 列), 不走 LDS.
    constexpr int LDS_STRIDE = 144;          // 16B 对齐 (int4 访问)
    constexpr int LDS_BUF = MT * 16 * LDS_STRIDE; // 每缓冲字节数
    const int lane_r = lane_row;             // 0..15
    const int lane_c = lane_col;             // 0..3
    const bool loader = (threadIdx.x < 64);  // 仅 warp0 做 A load

    // n_groups <= 2: 两个缓冲正好装全部组, 一次性写满后无需循环内屏障
    if (loader && n_groups <= 2) {
        #pragma unroll
        for(int g = 0; g < 2; ++g){
            if (g >= n_groups) break;
            #pragma unroll
            for(int mt = 0; mt < MT; ++mt){
                const Element* pa = A_ptr + row_a_arr[mt] + g * 128;
                int4 v0 = *(const int4*)(pa);
                int4 v1 = *(const int4*)(pa + 64);
                char* d = a_smem + g * LDS_BUF + mt * 16 * LDS_STRIDE
                        + lane_r * LDS_STRIDE + lane_c * 16;
                *reinterpret_cast<int4*>(d) = v0;
                *reinterpret_cast<int4*>(d + 64) = v1;
            }
        }
    }
    // 预取组 0 → 缓冲 0
    if (loader && n_groups > 2) {
        #pragma unroll
        for(int mt = 0; mt < MT; ++mt){
            const Element* pa = A_ptr + row_a_arr[mt];
            int4 v0 = *(const int4*)(pa);
            int4 v1 = *(const int4*)(pa + 64);
            char* d = a_smem + mt * 16 * LDS_STRIDE
                    + lane_r * LDS_STRIDE + lane_c * 16;
            *reinterpret_cast<int4*>(d) = v0;
            *reinterpret_cast<int4*>(d + 64) = v1;
        }
    }
    int4 nb[2];
    {
        const Element* pb = B_ptr + row_b;
        nb[0] = *(const int4*)(pb);
        nb[1] = *(const int4*)(pb + 64);
    }
    __syncthreads();

    for(int g = 0; g < n_groups; ++g){
        const int cur = (g & 1) * LDS_BUF;
        const int nxt = ((g & 1) ^ 1) * LDS_BUF;
        intx4 c[MT];
        #pragma unroll
        for(int mt = 0; mt < MT; ++mt) c[mt] = intx4{0,0,0,0};

        // 预取组 g+1 → 另一缓冲 (屏障在上轮末尾/本函数入口已保证
        // 该缓冲不再被读)
        if (n_groups > 2 && g + 1 < n_groups && loader) {
            const int gn = (g + 1) * 128;
            #pragma unroll
            for(int mt = 0; mt < MT; ++mt){
                const Element* pa = A_ptr + row_a_arr[mt] + gn;
                int4 v0 = *(const int4*)(pa);
                int4 v1 = *(const int4*)(pa + 64);
                char* d = a_smem + nxt + mt * 16 * LDS_STRIDE
                        + lane_r * LDS_STRIDE + lane_c * 16;
                *reinterpret_cast<int4*>(d) = v0;
                *reinterpret_cast<int4*>(d + 64) = v1;
            }
        }
        int4 tb[2];
        {
            const Element* pb = B_ptr + row_b + (g + 1 < n_groups ? (g + 1) : g) * 128;
            tb[0] = *(const int4*)(pb);
            tb[1] = *(const int4*)(pb + 64);
        }

        // 当前组: A 从 LDS 读 (组 g 的数据在 as[cur])
        #pragma unroll
        for(int mt = 0; mt < MT; ++mt){
            const char* base = a_smem + cur + mt * 16 * LDS_STRIDE
                             + lane_r * LDS_STRIDE + lane_c * 16;
            int4 la[2];
            la[0] = *reinterpret_cast<const int4*>(base);
            la[1] = *reinterpret_cast<const int4*>(base + 64);
            const int* ai = (const int*)&la[0];   // 32B 连续
            const int* bi = (const int*)&nb[0];
            #pragma unroll
            for(int j = 0; j < 4; ++j){
                int2 va = make_int2(ai[2*j], ai[2*j+1]);
                int2 vb = make_int2(bi[2*j], bi[2*j+1]);
                c[mt] = mmac<Element>(
                    *(typename vec<Element,8>::type*)&va,
                    *(typename vec<Element,8>::type*)&vb,
                    c[mt]);
            }
        }
        #pragma unroll
        for(int mt = 0; mt < MT; ++mt){
            float a_s = *(a_scale_rows[mt] + g);
            float w = MUL_TOPK ? topk_w_arr[mt] : 1.0f;
            #pragma unroll
            for(int i = 0; i < 4; ++i) {
                float b_s = *(b_scale_base + (b_scale_noff + lane_col + i * 4) * bsn + g);
                acc[mt][i] += (float)c[mt][i] * a_s * b_s * w;
            }
        }
        nb[0] = tb[0]; nb[1] = tb[1];
        if (n_groups > 2) __syncthreads();   // 本轮读结束, 下轮可覆写本缓冲
    }
}
// TILE16B_END
// DECODE kernel (UP / DOWN 共用, MUL_TOPK 模板参数区分)
// grid: (1, DIVIDE(N, BLOCK_N), DIVIDE(sorted_len, BLOCK_M))
// A tile: WARP_M=16 (1 tile), B tile: WARP_N=16, warp_n 沿 BLOCK_N 展开
// ---------------------------------------------------------------------------
template<typename Element, typename OutputType,
         int BLOCK_M, int BLOCK_N, bool MUL_TOPK, int MT_V>
__global__ void __launch_bounds__(512) moe_g128_decode(
    const Element* __restrict__ input,
    const Element* __restrict__ qweight,
    OutputType* __restrict__ output,
    const float* __restrict__ input_scale,
    const float* __restrict__ weight_scale,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t stride_asm,     // a_scale: [*, K/128] → stride(0) = K/128
    uint32_t stride_ask,
    uint32_t stride_bse,     // b_scale: [E, N, K/128] → stride(0) = N*K/128
    uint32_t stride_bsn,     // stride(1) = K/128
    uint32_t stride_bsk,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t delta,
    uint32_t num_experts,
    uint32_t num_pairs)
{
    const int bidx = blockIdx.z;   // 64-pair 块
    const int bidy = blockIdx.y;   // n 方向 block
    const int block_base = bidx * BLOCK_M * MT_V;

    if (block_base >= num_tokens_post_pad[0]) return;
    if (sorted_token_ids[block_base] >= int32_t(size_m * top_k)) return;

    const int32_t expert_id = expert_ids[bidx];
    // moe_align_block_size 的 expert_ids 尾部 (num_post 之后的 block) 未
    // 初始化 — 多余 grid 块必须按无效块早退, 否则垃圾 expert_id 导致
    // qweight 越界读 (VM fault)
    if (expert_id == -1 || expert_id < 0 || expert_id >= int32_t(num_experts)) return;

    constexpr int WARP_N = 16;
    constexpr int WARP_M = 16;
    constexpr int WARP_NUM = BLOCK_N / WARP_N;     // warp_n 数 (warp_k_num = 1)
    const int warp_id = threadIdx.x / 64;
    const int lane_id = threadIdx.x & 63;
    const int row_id = lane_id % 16;
    const int col_id = lane_id / 16;
    const int warp_n_id = warp_id;                 // n 方向 warp

    const int n_groups = size_k / G128;
    const uint64_t expert_off = ((uint64_t)size_n) * size_k * expert_id;
    const Element* B_base = qweight + expert_off;

    // 该 lane 的 B 行: n = bidy*BLOCK_N + warp_n*WARP_N + row_id
    const int n_row = bidy * BLOCK_N + warp_n_id * WARP_N + row_id;
    const int row_b = n_row * size_k + col_id * 16;

    // MT_V 个 16-token m-tile: sid/token/scale/weight 一次读入
    int sid_arr[MT_V];
    int row_a_arr[MT_V];
    const float* a_scale_rows[MT_V];
    float topk_w_arr[MT_V];
    #pragma unroll
    for(int mt = 0; mt < MT_V; ++mt) topk_w_arr[mt] = 1.f;
    #pragma unroll
    for(int mt = 0; mt < MT_V; ++mt){
        const int s = sorted_token_ids[std::min(block_base + mt * 16 + row_id,
                                               int(sorted_token_lens) - 1)];
        sid_arr[mt] = s;
        const bool ok = (s >= 0) && (s < int32_t(size_m * top_k))
                      && (!MUL_TOPK || s < int32_t(size_m));
        const int tok = ok ? (s / int(top_k)) : 0;
        row_a_arr[mt] = tok * int(size_k) + col_id * 16;
        a_scale_rows[mt] = input_scale + tok * int(stride_asm);
        if (MUL_TOPK)
            topk_w_arr[mt] = (s >= 0 && s < int32_t(size_m)) ? topk_weights[s] : 1.0f;
    }
    const int n_row_h1 = bidy * BLOCK_N + warp_n_id * WARP_N + row_id;
    const int row_b_h1 = n_row_h1 * size_k + col_id * 16;
    floatx4 acc_arr[MT_V];
    #pragma unroll
    for(int mt = 0; mt < MT_V; ++mt) acc_arr[mt] = floatx4{0.f, 0.f, 0.f, 0.f};
    __shared__ __align__(16) char a_smem[2 * MT_V * 16 * 144];
    // n_groups < 8 (如 gemm2 的 K=256 → 2 组): 屏障开销 > A 复用收益,
    // 走直连 load 版; 组多时走 LDS-A 版 (A 流量 ÷4).
    // gemm2 (MUL_TOPK) 小 pair 段 direct 更快 (g2prod 实测 0.94 vs
    // LDS 1.13); gemm1 阈值 256 (e2e 全 M 无慢点).
    const int tile_thr = MUL_TOPK ? 1024 : 256;
    if (num_pairs <= tile_thr) {
        g128_tile_direct<Element, MUL_TOPK, MT_V>(
            input, B_base, a_scale_rows,
            weight_scale + (uint64_t)expert_id * stride_bse,
            bidy * BLOCK_N + warp_n_id * WARP_N, (int)stride_bsn,
            n_groups, row_a_arr, row_b_h1, topk_w_arr,
            acc_arr, col_id);
    } else {
        g128_tile<Element, MUL_TOPK, MT_V>(
            input, B_base, a_scale_rows,
            weight_scale + (uint64_t)expert_id * stride_bse,
            bidy * BLOCK_N + warp_n_id * WARP_N, (int)stride_bsn,
            n_groups, row_a_arr, row_b_h1, topk_w_arr,
            acc_arr, row_id, col_id, a_smem);
    }

    // ---------------- 写回: 直接 scatter (4 分量 bf16) ----------------
    #pragma unroll
    for(int mt = 0; mt < MT_V; ++mt) {
        int32_t token_index = sid_arr[mt];
        // 写回界: UP (gemm1) pair 域 = size_m*top_k; DOWN (gemm2)
        // size_m 已是 pair 数 (= align numel), 界 = size_m.
        // align 的 fuse_fill 把 pad 项填成 topk_numel — DOWN 语义下
        // 该值 < size_m*top_k, 必须按 sid < size_m 判无效, 否则越界写.
        if (token_index >= 0 && token_index <
                (MUL_TOPK ? int32_t(size_m) : int32_t(size_m * top_k))) {
            #pragma unroll
            for(int i = 0; i < 4; ++i)
                output[(int64_t)token_index * size_n + bidy * BLOCK_N
                       + warp_n_id * WARP_N + col_id + i * 4] =
                    f32_to_output<OutputType>(acc_arr[mt][i]);
        }
    }
}

// ---------------------------------------------------------------------------
// C++ 入口: 签名与 moe_gemm_w8a8 完全一致
// ---------------------------------------------------------------------------
torch::Tensor moe_gemm_w8a8_group128(torch::Tensor input,
                                     torch::Tensor b_qweight,
                                     torch::Tensor output,
                                     torch::Tensor a_scale,
                                     torch::Tensor b_scale,
                                     std::optional<torch::Tensor> topk_weights,
                                     torch::Tensor sorted_token_ids,
                                     torch::Tensor expert_ids,
                                     torch::Tensor num_tokens_post_pad,
                                     int64_t top_k,
                                     int64_t mode,
                                     int64_t delta)
{
    const int size_m  = input.size(0);
    const int size_k  = input.size(1);
    // qweight [E, N, K] 未重排
    TORCH_CHECK(b_qweight.dim() == 3 && b_qweight.size(2) == size_k,
                "g128: expect qweight [E, N, K] with K == input.size(1)");
    const int EXPERTS = b_qweight.size(0);
    const int size_n  = b_qweight.size(1);
    const int topk_size = output.size(1);

    const int stride_asm = a_scale.stride(0);
    const int stride_ask = a_scale.stride(1);
    const int stride_bse = b_scale.stride(0);
    const int stride_bsn = b_scale.stride(1);
    const int stride_bsk = b_scale.stride(2);

    TORCH_CHECK(size_k % 128 == 0, "g128: K must be multiple of 128, got ", size_k);

    const bool first_stage = !topk_weights.has_value();
    const float* topk_w_ptr = first_stage ? nullptr
        : (const float*)topk_weights.value().data_ptr();

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 128;
    // BLOCK_M=64: 每 warp 4 个 16-token m-tile (MT=4), B 数据复用 4×

    auto stream = at::cuda::getCurrentHIPStream();
    const int64_t EM = sorted_token_ids.size(0);

    // 无效 m 块: 用 num_tokens_post_pad 在 kernel 内早退; grid 直接按
    // sorted 长度开, 与既有 kernel 相同 (多余块早退, 无副作用)

    // 分派: 短序列 (decode/小中 batch) 用 MT=1 (16-pair 块, 并行度高);
    // 长序列 (prefill) 用 MT=4 (64-pair 块, B 复用 4x + LDS-A).
    // 调用方需按所选块大小提供匹配的 align (block_size 16 / 64).
    const int64_t n_pairs = (int64_t)output.numel() / size_n;
    // 大 batch 用 MT=4 (64-pair 块): gemm1 与 gemm2(top_k=1, A 为 pair
    // 行) 均安全 — 块内 4 个 16-tile 对应连续 pair, token 映射无碰撞.
    // 调用方须提供匹配块大小的 align (block_size 16 / 64).
    const bool use_mt4 = (n_pairs > 1024);
    const int mt_blk = use_mt4 ? 4 : 1;
    dim3 grid(1, DIVIDE(size_n, BLOCK_N), DIVIDE(EM, BLOCK_M * mt_blk));
    dim3 block((BLOCK_N / 16) * 64, 1, 1);

    if (output.scalar_type() == at::ScalarType::BFloat16) {
        if (first_stage) {
            if (use_mt4)
                moe_g128_decode<char, bhalf_t, 16, BLOCK_N, false, 4><<<grid, block, 0, stream>>>(
                (const char*)input.data_ptr<int8_t>(),
                (const char*)b_qweight.data_ptr<int8_t>(),
                (bhalf_t*)output.data_ptr(),
                (const float*)a_scale.data_ptr(),
                (const float*)b_scale.data_ptr(),
                topk_w_ptr,
                sorted_token_ids.data_ptr<int32_t>(),
                expert_ids.data_ptr<int32_t>(),
                num_tokens_post_pad.data_ptr<int32_t>(),
                size_m, size_n, size_k,
                stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk,
                (uint32_t)EM, (uint32_t)top_k, (uint32_t)delta, (uint32_t)EXPERTS,
                (uint32_t)(output.size(0) * topk_size));
            else
                moe_g128_decode<char, bhalf_t, 16, BLOCK_N, false, 1><<<grid, block, 0, stream>>>(
                (const char*)input.data_ptr<int8_t>(),
                (const char*)b_qweight.data_ptr<int8_t>(),
                (bhalf_t*)output.data_ptr(),
                (const float*)a_scale.data_ptr(),
                (const float*)b_scale.data_ptr(),
                topk_w_ptr,
                sorted_token_ids.data_ptr<int32_t>(),
                expert_ids.data_ptr<int32_t>(),
                num_tokens_post_pad.data_ptr<int32_t>(),
                size_m, size_n, size_k,
                stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk,
                (uint32_t)EM, (uint32_t)top_k, (uint32_t)delta, (uint32_t)EXPERTS,
                (uint32_t)(output.size(0) * topk_size));
        } else {
            if (use_mt4)
                moe_g128_decode<char, bhalf_t, 16, BLOCK_N, true, 4><<<grid, block, 0, stream>>>(
                (const char*)input.data_ptr<int8_t>(),
                (const char*)b_qweight.data_ptr<int8_t>(),
                (bhalf_t*)output.data_ptr(),
                (const float*)a_scale.data_ptr(),
                (const float*)b_scale.data_ptr(),
                topk_w_ptr,
                sorted_token_ids.data_ptr<int32_t>(),
                expert_ids.data_ptr<int32_t>(),
                num_tokens_post_pad.data_ptr<int32_t>(),
                size_m, size_n, size_k,
                stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk,
                (uint32_t)EM, (uint32_t)top_k, (uint32_t)delta, (uint32_t)EXPERTS,
                (uint32_t)(output.size(0) * topk_size));
            else
                moe_g128_decode<char, bhalf_t, 16, BLOCK_N, true, 1><<<grid, block, 0, stream>>>(
                (const char*)input.data_ptr<int8_t>(),
                (const char*)b_qweight.data_ptr<int8_t>(),
                (bhalf_t*)output.data_ptr(),
                (const float*)a_scale.data_ptr(),
                (const float*)b_scale.data_ptr(),
                topk_w_ptr,
                sorted_token_ids.data_ptr<int32_t>(),
                expert_ids.data_ptr<int32_t>(),
                num_tokens_post_pad.data_ptr<int32_t>(),
                size_m, size_n, size_k,
                stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk,
                (uint32_t)EM, (uint32_t)top_k, (uint32_t)delta, (uint32_t)EXPERTS,
                (uint32_t)(output.size(0) * topk_size));
        }
    } else if (output.scalar_type() == at::ScalarType::Half) {
        TORCH_CHECK(false, "g128: half output not implemented yet");
    } else {
        TORCH_CHECK(false, "g128: unsupported output dtype");
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "moe_gemm_w8a8_group128 launch failed: ",
                cudaGetErrorString(err));
    return output;
}

} // namespace moe_g128
} // namespace at::native

namespace at::native { namespace moe_g128 {
void g128_dbg_clear_host() {
    g128_dbg_clear<<<1, 256, 0, at::cuda::getCurrentHIPStream()>>>();
}
int64_t g128_dbg_addr() { return (int64_t)g128_dbg_buf; }
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_gemm_w8a8_group128",
          &at::native::moe_g128::moe_gemm_w8a8_group128,
          "INT8 W8A8 group-128 MoE GEMM (decode)");
    m.def("g128_dbg_clear_host", &at::native::moe_g128::g128_dbg_clear_host);
    m.def("g128_dbg_addr", &at::native::moe_g128::g128_dbg_addr);
}
