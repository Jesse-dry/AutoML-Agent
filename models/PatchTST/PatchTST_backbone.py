"""
PatchTST 主干模型
================
将时序切分为 overlapping patches，通过通道独立 Transformer 编码，
最后展平投影到预测窗口。配合 RevIN 处理分布偏移。

输入形状: [bs, nvars, seq_len]  (batch_first, 通道优先)
输出形状: [bs, nvars, pred_len]

架构总览：
  1. RevIN norm      — 逐通道归一化
  2. Patching        — unfold 切分 + 可选 stride padding
  3. TSTiEncoder     — 通道独立 Transformer
  4. Flatten_Head    — 展平 + 线性投影
  5. RevIN denorm    — 逆向还原

参考：Nie et al., ICLR 2023.
"""

import torch
import torch.nn as nn
from typing import Optional

from RevIN import RevIN
from PatchTST_layers import TSTiEncoder, Flatten_Head


class PatchTST_backbone(nn.Module):
    """
    PatchTST 主干网络。

    Parameters
    ----------
    c_in : int
        输入通道数（变量数）
    context_window : int
        输入序列长度（历史窗口）
    target_window : int
        预测序列长度（预测步长）
    patch_len : int
        每个 patch 的长度
    stride : int
        patch 之间的步长
    max_seq_len : int
        位置编码的最大长度
    n_layers : int
        Transformer encoder 层数
    d_model : int
        模型隐藏维度
    n_heads : int
        注意力头数
    d_k, d_v : int | None
        每头的 key/value 维度（默认 d_model // n_heads）
    d_ff : int
        前馈网络隐藏层维度
    dropout : float
        残差 dropout
    attn_dropout : float
        注意力 dropout
    head_dropout : float
        预测头 dropout
    activation : str
        激活函数 ("gelu" / "relu")
    res_attention : bool
        是否启用残差注意力
    pre_norm : bool
        True → pre-layer normalization
    pe : str
        位置编码类型 ("zeros", "sincos", "normal", "uniform")
    learn_pe : bool
        位置编码是否可学习
    revin : bool
        是否使用 RevIN
    revin_affine : bool
        RevIN 是否使用可学习仿射参数
    individual : bool
        是否每通道独立预测头
    padding_patch : str | None
        "end" → 在序列末尾补 stride 长度以保证覆盖最后的数据点
    """

    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        max_seq_len: int = 1024,
        # Encoder
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        head_dropout: float = 0.0,
        activation: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        # RevIN
        revin: bool = True,
        revin_affine: bool = True,
        # Head
        individual: bool = False,
        # Patching
        padding_patch: Optional[str] = None,
    ):
        super().__init__()

        # --- 记录关键参数 ---
        self.c_in = c_in
        self.context_window = context_window
        self.target_window = target_window
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.revin = revin
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.head_dropout = head_dropout
        self.activation = activation
        self.res_attention = res_attention
        self.pre_norm = pre_norm
        self.pe = pe
        self.learn_pe = learn_pe
        self.individual = individual

        # 计算 patch 数量
        self.patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":
            self.patch_num += 1

        # --- RevIN ---
        if revin:
            self.revin_layer = RevIN(
                c_in,
                affine=revin_affine,
                subtract_last=False,
            )

        # --- Backbone Encoder ---
        self.backbone = TSTiEncoder(
            c_in=c_in,
            patch_num=self.patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation=activation,
            res_attention=res_attention,
            pre_norm=pre_norm,
            pe=pe,
            learn_pe=learn_pe,
        )

        # --- Head ---
        self.head = Flatten_Head(
            n_vars=c_in,
            d_model=d_model,
            patch_num=self.patch_num,
            target_window=target_window,
            head_dropout=head_dropout,
            individual=individual,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : [bs, nvars, seq_len]

        Returns
        -------
        [bs, nvars, target_window]
        """
        bs, nvars, seq_len = z.shape

        # ---- Step 1: RevIN norm ----
        if self.revin:
            # RevIN 期望 [bs, seq_len, nvars]
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, mode="norm")
            z = z.permute(0, 2, 1)  # 回到 [bs, nvars, seq_len]

        # ---- Step 2: Patching ----
        # unfold 沿最后一维切 patch: [bs, nvars, seq_len] → [bs, nvars, patch_num, patch_len]
        if self.padding_patch == "end":
            # 在末尾补 stride 长度
            z = nn.functional.pad(z, (0, self.stride))
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # unfold 输出 shape: [bs, nvars, patch_num, patch_len]
        z = z.permute(0, 1, 3, 2)  # → [bs, nvars, patch_len, patch_num]

        # ---- Step 3: Backbone Encoder ----
        z = self.backbone(z)  # → [bs, nvars, d_model, patch_num]

        # ---- Step 4: Head Projection ----
        z = self.head(z)  # → [bs, nvars, target_window]

        # ---- Step 5: RevIN denorm ----
        if self.revin:
            z = z.permute(0, 2, 1)  # [bs, target_window, nvars]
            z = self.revin_layer(z, mode="denorm")
            z = z.permute(0, 2, 1)  # 回到 [bs, nvars, target_window]

        return z
