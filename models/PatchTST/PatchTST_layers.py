"""
PatchTST 核心层组件
===================
包含：
  - Transpose 辅助模块
  - 位置编码生成函数
  - _MultiheadAttention + _ScaledDotProductAttention
  - TSTEncoderLayer / TSTEncoder（通道独立 Transformer 编码器）
  - TSTiEncoder（含线性投影 + 位置编码的输入编码器）
  - Flatten_Head（预测头）

参考：Nie et al., "A Time Series is Worth 64 Words:
      Long-term Forecasting with Transformers", ICLR 2023.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ================================================================
# 辅助模块
# ================================================================


class Transpose(nn.Module):
    """可选的 contiguous 转置包装。"""

    def __init__(self, *dims: int, contiguous: bool = False):
        super().__init__()
        self.dims = dims
        self.contiguous_flag = contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.contiguous_flag:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


def get_activation_fn(activation: str) -> nn.Module:
    """根据字符串返回激活函数模块。"""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"不支持的激活函数: {activation}")


# ================================================================
# 位置编码
# ================================================================


def positional_encoding(
    pe: str,
    learn_pe: bool,
    q_len: int,
    d_model: int,
) -> nn.Parameter:
    """
    生成可学习/固定的位置编码。

    Parameters
    ----------
    pe : str
        编码类型：'zeros', 'normal', 'uniform', 'sincos' 等
    learn_pe : bool
        是否可学习
    q_len : int
        序列长度
    d_model : int
        模型维度

    Returns
    -------
    nn.Parameter, shape [q_len, d_model]
    """
    if pe in ("None", "zero", "zeros"):
        pe_tensor = torch.zeros(q_len, d_model)
    elif pe in ("normal", "gauss"):
        pe_tensor = torch.randn(q_len, d_model)
    elif pe == "uniform":
        pe_tensor = torch.empty(q_len, d_model).uniform_(-0.02, 0.02)
    elif pe == "sincos":
        pe_tensor = _sinusoidal_position_encoding(q_len, d_model)
    else:
        raise ValueError(f"未知的位置编码类型: {pe}")

    return nn.Parameter(pe_tensor, requires_grad=learn_pe)


def _sinusoidal_position_encoding(q_len: int, d_model: int) -> torch.Tensor:
    """标准正弦余弦位置编码。"""
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ================================================================
# 注意力机制
# ================================================================


class _ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力。

    支持：
      - 注意力 mask（布尔型 float 型均可）
      - key_padding_mask
      - 残差注意力连接（Realformer 风格，prev 参数）
    """

    def __init__(self, d_k: int, lsa: bool = False):
        super().__init__()
        self.d_k = d_k
        self.lsa = lsa
        # 可学习的温度缩放（Learnable Scaled Attention）
        if lsa:
            self.scale = nn.Parameter(torch.tensor(d_k ** -0.5))
        else:
            self.scale = d_k ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prev: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        q : [bs, n_heads, q_len, d_k]
        k : [bs, n_heads, d_k, seq_len]
        v : [bs, n_heads, seq_len, d_v]
        prev : [bs, n_heads, q_len, seq_len] or None
            前一层注意力分数（残差注意力）
        attn_mask : [q_len, seq_len] or None
        key_padding_mask : [bs, seq_len] or None

        Returns
        -------
        attn_output : [bs, n_heads, q_len, d_v]
        """
        # 计算注意力分数
        attn_scores = torch.matmul(q, k) * self.scale  # [bs, n_heads, q_len, seq_len]

        if prev is not None:
            attn_scores = attn_scores + prev

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_scores += attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: [bs, seq_len] → [bs, 1, 1, seq_len]
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=0.0, training=self.training)
        output = torch.matmul(attn_weights, v)  # [bs, n_heads, q_len, d_v]
        return output


class _MultiheadAttention(nn.Module):
    """
    多头注意力（自定义实现，支持高效 K 转置）。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        dropout: float = 0.0,
        lsa: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.d_v = d_v if d_v is not None else d_model // n_heads

        self.W_Q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_K = nn.Linear(d_model, n_heads * self.d_k)
        self.W_V = nn.Linear(d_model, n_heads * self.d_v)

        self.out_proj = nn.Linear(n_heads * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.attention = _ScaledDotProductAttention(self.d_k, lsa=lsa)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prev: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        q, k, v : [bs, seq_len, d_model]
        prev : [bs, n_heads, q_len, seq_len] or None
        attn_mask : [q_len, seq_len] or None
        key_padding_mask : [bs, seq_len] or None

        Returns
        -------
        [bs, seq_len, d_model]
        """
        bs, q_len = q.shape[:2]
        seq_len = k.shape[1]

        # 投影
        Q = self.W_Q(q).view(bs, q_len, self.n_heads, self.d_k).transpose(1, 2)
        # [bs, n_heads, q_len, d_k]

        K = self.W_K(k).view(bs, seq_len, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        # [bs, n_heads, d_k, seq_len] — 预转置方便 matmul

        V = self.W_V(v).view(bs, seq_len, self.n_heads, self.d_v).transpose(1, 2)
        # [bs, n_heads, seq_len, d_v]

        # 注意力
        attn_out = self.attention(Q, K, V, prev=prev,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask)
        # [bs, n_heads, q_len, d_v]

        # 合并头
        attn_out = attn_out.transpose(1, 2).contiguous().view(bs, q_len, -1)
        # [bs, q_len, n_heads * d_v]

        return self.dropout(self.out_proj(attn_out))


# ================================================================
# Transformer Encoder Layers
# ================================================================


class TSTEncoderLayer(nn.Module):
    """
    单层 Transformer Encoder：Self-Attention + Feed-Forward。

    可选 pre_norm 和残差注意力。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        activation: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
    ):
        super().__init__()
        self.res_attention = res_attention
        self.pre_norm = pre_norm

        # LayerNorm / BatchNorm 选择 — 这里用 LayerNorm（更通用）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = _MultiheadAttention(
            d_model, n_heads, d_k, d_v,
            dropout=attn_dropout,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        src: torch.Tensor,
        prev: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Parameters
        ----------
        src : [bs, seq_len, d_model]
        prev : 前一层注意力分数
        attn_mask, key_padding_mask : 注意力 mask

        Returns
        -------
        (output, attn_scores) 其中 output: [bs, seq_len, d_model]
        """
        # --- Self-Attention 子层 ---
        if self.pre_norm:
            src2 = self.norm1(src)
            attn_out, attn_scores = self.self_attn(
                src2, src2, src2, prev=prev,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            ), None
            src = src + self.dropout1(attn_out)
        else:
            attn_out = self.self_attn(
                src, src, src, prev=prev,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            # attn_out 不返回 scores，我们单独处理（简化）
            src = self.norm1(src + self.dropout1(attn_out))

        # --- Feed-Forward 子层 ---
        if self.pre_norm:
            src2 = self.norm2(src)
            ff_out = self.ff(src2)
            src = src + self.dropout2(ff_out)
        else:
            ff_out = self.ff(src)
            src = self.norm2(src + self.dropout2(ff_out))

        return src, None  # 简化：不传递注意力分数


class TSTEncoder(nn.Module):
    """堆叠 N 个 TSTEncoderLayer。"""

    def __init__(self, encoder_layer: TSTEncoderLayer, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            encoder_layer for _ in range(n_layers)
        ])

    def forward(
        self,
        src: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src : [bs, seq_len, d_model]

        Returns
        -------
        [bs, seq_len, d_model]
        """
        output = src
        for layer in self.layers:
            output, _ = layer(
                output,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        return output


# ================================================================
# TSTiEncoder — 通道独立编码器
# ================================================================


class TSTiEncoder(nn.Module):
    """
    通道独立（Channel-Independent）时间序列 Transformer 编码器。

    每个变量独立通过同一个 Transformer encoder，
    将 (bs, nvars, patch_num, patch_len) 映射为 (bs, nvars, d_model, patch_num)。

    关键设计：
      - W_P:  把每个 patch_len 长的 patch 线性投影到 d_model
      - W_pos: 位置编码（patch 级别）
      - encoder: 共享的 TSTEncoder
    """

    def __init__(
        self,
        c_in: int,
        patch_num: int,
        patch_len: int,
        max_seq_len: int = 1024,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        activation: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
    ):
        super().__init__()

        # --- Patch 投影 ---
        self.W_P = nn.Linear(patch_len, d_model)

        # --- 位置编码 ---
        self.seq_len = patch_num
        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)

        # --- 残差 dropout ---
        self.dropout = nn.Dropout(dropout)

        # --- Transformer Encoder ---
        encoder_layer = TSTEncoderLayer(
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
        )
        self.encoder = TSTEncoder(encoder_layer, n_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [bs, nvars, patch_len, patch_num]
            其中 patch_num = 已分 patch 的数量

        Returns
        -------
        [bs, nvars, d_model, patch_num]
        """
        bs, nvars, patch_len, patch_num = x.shape

        # 1. 转置 → [bs, nvars, patch_num, patch_len]
        x = x.permute(0, 1, 3, 2)

        # 2. 线性投影 → [bs, nvars, patch_num, d_model]
        x = self.W_P(x)

        # 3. 通道独立：合并 batch 和 nvars → [bs * nvars, patch_num, d_model]
        x = x.reshape(bs * nvars, patch_num, -1)

        # 4. 加位置编码 + dropout
        x = x + self.W_pos.unsqueeze(0)  # [1, patch_num, d_model]
        x = self.dropout(x)

        # 5. Transformer 编码
        x = self.encoder(x)  # [bs * nvars, patch_num, d_model]

        # 6. 恢复为 [bs, nvars, patch_num, d_model]
        x = x.reshape(bs, nvars, patch_num, -1)

        # 7. 转置 → [bs, nvars, d_model, patch_num]
        x = x.permute(0, 1, 3, 2)

        return x


# ================================================================
# Flatten_Head — 预测头
# ================================================================


class Flatten_Head(nn.Module):
    """
    将编码器输出展平后线性投影到预测窗口。

    [bs, nvars, d_model, patch_num] → [bs, nvars, target_window]

    Parameters
    ----------
    individual : bool
        True  → 每个通道独立的投影头
        False → 所有通道共享一个投影头
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        patch_num: int,
        target_window: int,
        head_dropout: float = 0.0,
        individual: bool = False,
    ):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        self.head_dim = d_model * patch_num

        if individual:
            # 每通道一个独立 Linear
            self.linears = nn.ModuleList([
                nn.Linear(self.head_dim, target_window) for _ in range(n_vars)
            ])
        else:
            self.linear = nn.Linear(self.head_dim, target_window)

        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [bs, nvars, d_model, patch_num]

        Returns
        -------
        [bs, nvars, target_window]
        """
        # 展平后两维 → [bs, nvars, d_model * patch_num]
        x = x.flatten(start_dim=-2)

        if self.individual:
            # 逐通道投影 → stack
            outputs = []
            for i in range(self.n_vars):
                out_i = self.linears[i](x[:, i, :])  # [bs, target_window]
                outputs.append(out_i)
            x = torch.stack(outputs, dim=1)  # [bs, nvars, target_window]
        else:
            x = self.linear(x)  # [bs, nvars, target_window]

        return self.dropout(x)
