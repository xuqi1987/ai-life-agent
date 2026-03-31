"""内置工具集 — 提供开箱即用的基础工具。

包含以下工具：
- **calculator**: 安全的数学计算器（防注入）
- **get_current_time**: 获取当前日期和时间
- **echo**: 原样返回输入（测试用）

使用示例::

    from ai_life_agent.tools.builtin import register_builtin_tools
    from ai_life_agent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    register_builtin_tools(registry)

    print(registry.execute("calculator", {"expression": "2 + 3 * 4"}))  # "14"
    print(registry.execute("get_current_time", {}))                      # "2026-03-31 14:00:00"
"""

from __future__ import annotations

import math
from datetime import datetime

from ai_life_agent.tools.registry import ToolParameter, ToolRegistry


def register_builtin_tools(registry: ToolRegistry) -> None:
    """将所有内置工具注册到指定注册表。

    Args:
        registry: 目标 ToolRegistry 实例。
    """
    registry.register_tool(
        name="calculator",
        description=(
            "安全的数学计算器。支持加减乘除、幂次、取余等基本运算，"
            "以及 sin/cos/sqrt/log 等数学函数。"
            "示例: '2 + 3 * 4', 'sqrt(16)', 'sin(pi/2)', 'log(e)'."
        ),
        parameters=[
            ToolParameter(
                name="expression",
                description="数学表达式字符串，例如 '2 + 3 * 4' 或 'sqrt(16)'",
                type="string",
                required=True,
            )
        ],
        func=_calculator,
    )

    registry.register_tool(
        name="get_current_time",
        description="获取当前的日期和时间。",
        parameters=[
            ToolParameter(
                name="format",
                description="时间格式字符串，默认为 '%Y-%m-%d %H:%M:%S'",
                type="string",
                required=False,
            )
        ],
        func=_get_current_time,
    )

    registry.register_tool(
        name="echo",
        description="原样返回输入的文本（主要用于调试和测试）。",
        parameters=[
            ToolParameter(
                name="text",
                description="需要原样返回的文本",
                type="string",
                required=True,
            )
        ],
        func=_echo,
    )


def _calculator(expression: str) -> str:
    """安全计算数学表达式。

    只允许数学运算，不允许任何危险代码执行，防止注入攻击。

    Args:
        expression: 数学表达式字符串，例如 '2+3*4' 或 'sqrt(16)'。

    Returns:
        计算结果字符串，或错误描述字符串。
    """
    # 安全符号白名单（只暴露纯数学函数）
    allowed_names: dict[str, object] = {
        "abs": abs,
        "round": round,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "ceil": math.ceil,
        "floor": math.floor,
        "pow": math.pow,
        "factorial": math.factorial,
    }

    try:
        # 禁止双下划线（防止 __import__、__builtins__ 等攻击）
        if "__" in expression:
            raise ValueError("不允许使用 '__'")

        # 限制字符集（只允许数字、运算符、括号、小数点、空白和字母）
        allowed_chars = set("0123456789+-*/()., abcdefghijklmnopqrstuvwxyz_ \t\n")
        for char in expression:
            if char not in allowed_chars:
                raise ValueError(f"不允许的字符: {char!r}")

        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        # 格式化结果：整数去掉小数点，浮点数保留合理精度
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)

    except ZeroDivisionError:
        return "错误: 除数不能为零"
    except ValueError as e:
        return f"输入错误: {e}"
    except Exception as e:
        return f"计算错误: {e}"


def _get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """获取当前本地时间。

    Args:
        format: strftime 格式字符串，默认为 '%Y-%m-%d %H:%M:%S'。

    Returns:
        格式化的当前时间字符串。
    """
    try:
        return datetime.now().strftime(format)
    except Exception as e:
        return f"时间格式错误: {e}"


def _echo(text: str) -> str:
    """原样返回输入文本（用于调试）。

    Args:
        text: 需要返回的文本。

    Returns:
        与输入相同的文本。
    """
    return text
