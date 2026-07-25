"""
Reversible Instance Normalization (RevIN)
=========================================
对每个通道独立做 instance normalization，并在预测后逆向还原。

原理：
  - norm 阶段：减去均值除以标准差 + 可选仿射变换
  - denorm 阶段：逆向操作恢复原始分布

在 PatchTST 中的作用：
  消除训练/测试集之间的分布偏移（distribution shift），
  让模型关注时序模式本身而非量级波动。

Reference: Kim et al., "Reversible Instance Normalization for Accurate
           Time-Series Forecasting against Distribution Shift", ICLR 2022.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization。

    对输入 x 形状为 [bs, seq_len, nvars] 的时序数据：
      norm  → 逐样本逐通道归一化
      denorm → 逆向还原（用于预测输出）

    Parameters
    ----------
    num_features : int
        通道数 / 特征数
    eps : float
        防止除零的小常数
    affine : bool
        是否学习逐通道的 scale 和 bias
    subtract_last : bool
        True 时减去最后一个时间步而非均值（适合某些非平稳场景）
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self._init_params()

    def _init_params(self):
        """初始化可学习的仿射参数（逐通道）。"""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape [bs, seq_len, nvars]
        mode : str
            "norm"  — 归一化（输入到模型前调用）
            "denorm" — 逆归一化（模型输出后调用）

        Returns
        -------
        torch.Tensor, 同 shape
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"RevIN mode '{mode}' 不支持，只支持 'norm' / 'denorm'")
        return x

    def _get_statistics(self, x: torch.Tensor):
        """计算并存储每个样本每个通道的统计量。"""
        # x: [bs, seq_len, nvars]
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1).detach()  # [bs, 1, nvars]
        else:
            # 对 seq_len 维度归约，保留 bs 和 nvars
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()  # [bs, 1, nvars]
        # 标准差（有偏估计，匹配论文）
        self.stdev = (
            torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
            .detach()
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """应用归一化。"""
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev

        if self.affine:
            # affine_weight/bias: [nvars] → [1, 1, nvars]
            x = x * self.affine_weight.view(1, 1, -1)
            x = x + self.affine_bias.view(1, 1, -1)
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """逆向还原。"""
        if self.affine:
            x = x - self.affine_bias.view(1, 1, -1)
            x = x / (self.affine_weight.view(1, 1, -1) + self.eps * self.eps)

        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
