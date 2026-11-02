# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""INT8 W8A8 group-128 MoE GEMM — 独立 JIT 扩展加载器 (新建文件).

不修改 lightop.op 的任何既有导出: 本模块用
torch.utils.cpp_extension.load 在运行时把
``csrc/moew8a8/moe_w8a8_group128.cu`` 编译为独立扩展 ``lightop_g128``。

用法::

    from lightop.group128_ext import moe_gemm_w8a8_group128
    out = moe_gemm_w8a8_group128(input, b_qweight, output, a_scale, b_scale,
                                 topk_weights, sorted_token_ids, expert_ids,
                                 num_tokens_post_pad, top_k, mode, delta)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

_EXT = None
_BUILD_DIR = os.environ.get(
    "LIGHTOP_G128_BUILD_DIR",
    os.path.join(Path.home(), ".cache", "lightop_g128"),
)


def _build():
    global _EXT
    if _EXT is not None:
        return _EXT

    from torch.utils.cpp_extension import load

    csrc = Path("/workspace/lightop/csrc/moew8a8")  # 仓库布局 (python/lightop 上两级即仓库根)
    src = csrc / "moe_w8a8_group128.cu"
    if not src.exists():
        # 兼容 pip 安装布局: 源文件随包 resources 打包时在 lightop/csrc 下
        alt = Path(__file__).resolve().parent / "csrc" / "moew8a8" / src.name
        if alt.exists():
            src = alt
        else:
            raise FileNotFoundError(
                f"moe_w8a8_group128.cu not found (tried {csrc} and packaged layout)")

    os.makedirs(_BUILD_DIR, exist_ok=True)
    _EXT = load(
        name="lightop_g128",
        sources=[str(src)],
        extra_include_paths=[str(csrc),   # moe_w8a8_utils_hip.h 等
                             str(csrc.parent)],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-w",
            *os.environ.get("LIGHTOP_G128_EXTRA", "").split(),
            "--gpu-max-threads-per-block=1024",
            # 与 lightop 主构建一致的架构集合
            *[f"--offload-arch={a}" for a in
              os.environ.get("PYTORCH_ROCM_ARCH", "gfx928;gfx936;gfx938").split(";")],
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=os.environ.get("LIGHTOP_G128_VERBOSE", "0") == "1",
        build_directory=_BUILD_DIR,
    )
    return _EXT


def moe_gemm_w8a8_group128(input: torch.Tensor,
                           b_qweight: torch.Tensor,
                           output: torch.Tensor,
                           a_scale: torch.Tensor,
                           b_scale: torch.Tensor,
                           topk_weights,
                           sorted_token_ids: torch.Tensor,
                           expert_ids: torch.Tensor,
                           num_tokens_post_pad: torch.Tensor,
                           top_k: int,
                           mode: int = 0,
                           delta: int = 1) -> torch.Tensor:
    """INT8 W8A8 group-128 MoE GEMM (decode).

    与 ``lightop.op.moe_gemm_w8a8`` 签名一致, 但要求:
      * ``b_qweight`` [E, N, K] int8 未重排 (K 为 128 的倍数)
      * ``b_scale``  [E, N, K/128] fp32
      * ``a_scale``  [*, K/128] fp32 (per-token-group)
    ``topk_weights`` 为 None 时走 gemm1 (UP), 有值时走 gemm2 (DOWN)。
    """
    ext = _build()
    return ext.moe_gemm_w8a8_group128(
        input, b_qweight, output, a_scale, b_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_pad,
        int(top_k), int(mode), int(delta))
