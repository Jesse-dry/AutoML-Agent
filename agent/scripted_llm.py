# ScriptedLLM：测试 / --dry-run 用的确定性 LLM
# ---------------------------------------------------------------
# chat() 按调用序号返回脚本提供的 v2 JSON 字符串（不走真实 API）。
# 与 QwenClient 鸭子类型兼容：.model / .dry_run / .chat(messages, temperature)。
#
# 用途：
#   - 复现"改善→回归→回滚→再改善"（E6 完成标准）——固定脚本 + seed 确定性
#   - 复现多候选择优（固定 3 候选，其中 1 个恒优）
# ---------------------------------------------------------------
from typing import Callable, List


class ScriptedLLM:
    """确定性 LLM 客户端。script: 轮次号 → v2 JSON 字符串。"""

    def __init__(self, script: Callable[[int], str]):
        self.script = script
        self._count = 0
        self.model = "scripted"
        self.dry_run = True
        self.last_response: str = ""

    def chat(self, messages, temperature: float = 0.3) -> str:
        self._count += 1
        resp = self.script(self._count)
        self.last_response = resp
        return resp


def sequence(jsons: List[str]) -> Callable[[int], str]:
    """把一串 v2 JSON 字符串编成脚本：第 i 次调用返回第 i 条，超出用最后一条。"""
    def _script(round_no: int) -> str:
        idx = min(round_no - 1, len(jsons) - 1)
        return jsons[idx]
    return _script
