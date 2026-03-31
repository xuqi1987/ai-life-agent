"""工具注册表 — 统一管理所有可被 Agent 调用的工具。

设计原则：
1. **工具即函数**：每个工具对应一个 Python 函数，降低接入成本
2. **自描述**：工具元数据（名称、描述、参数）通过装饰器声明
3. **可插拔**：工具可在运行时动态注册/注销
4. **类型安全**：基于 Python 类型注解自动生成 JSON Schema

使用示例::

    registry = ToolRegistry()

    # 方式一：装饰器注册
    @registry.register(description="计算两数之和")
    def add(a: int, b: int) -> int:
        \"\"\"将两个数字相加。\"\"\"
        return a + b

    # 方式二：显式注册
    registry.register_tool(
        name="echo",
        description="原样返回文本",
        parameters=[ToolParameter("text", "输入文本", "string")],
        func=lambda text: text,
    )

    # 获取 Anthropic 格式 Schema（传给 LLM）
    schemas = registry.get_schemas()

    # 执行工具
    result = registry.execute("add", {"a": 1, "b": 2})  # => 3
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """工具参数定义。

    Attributes:
        name: 参数名称（需与函数参数名一致）。
        description: 参数描述，展示给 LLM 以帮助理解如何传参。
        type: JSON Schema 类型，可选: "string", "number", "boolean", "array", "object"。
        required: 是否必填，默认 True。
        enum: 枚举值列表，限制参数只能取特定值。
    """

    name: str
    description: str
    type: str = "string"
    required: bool = True
    enum: list[Any] | None = None


@dataclass
class ToolDefinition:
    """工具定义 — 包含元数据和可执行函数。

    Attributes:
        name: 工具名称（全局唯一）。
        description: 工具功能描述，LLM 根据此决定是否调用。
        parameters: 参数列表。
        func: 实际执行函数。
        enabled: 是否启用，禁用后 LLM 不会看到此工具。
    """

    name: str
    description: str
    parameters: list[ToolParameter]
    func: Callable
    enabled: bool = True

    def to_anthropic_schema(self) -> dict[str, Any]:
        """转换为 Anthropic SDK 格式的工具 Schema。

        Anthropic 工具 Schema 格式::

            {
                "name": "tool_name",
                "description": "工具描述",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "..."},
                    },
                    "required": ["param1"],
                },
            }

        Returns:
            符合 Anthropic API 格式的工具定义字典。
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def execute(self, **kwargs: Any) -> Any:
        """执行工具函数。

        Args:
            **kwargs: 工具参数，键名与函数参数名一致。

        Returns:
            工具执行结果（任意类型）。

        Raises:
            RuntimeError: 工具被禁用时。
            Exception: 工具函数执行失败时透传原始异常。
        """
        if not self.enabled:
            raise RuntimeError(f"工具 '{self.name}' 已被禁用")
        logger.debug("执行工具: %s(%s)", self.name, kwargs)
        return self.func(**kwargs)


class ToolRegistry:
    """工具注册表 — 管理所有已注册工具的生命周期。

    线程安全说明：当前实现非线程安全，如需多线程注册，请自行加锁。

    示例::

        from ai_life_agent.tools.registry import ToolRegistry

        registry = ToolRegistry()

        @registry.register(description="将摄氏度转为华氏度")
        def celsius_to_fahrenheit(celsius: float) -> float:
            return celsius * 9 / 5 + 32

        print(registry.execute("celsius_to_fahrenheit", {"celsius": 100}))  # 212.0
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str | None = None,
        description: str = "",
        parameters: list[ToolParameter] | None = None,
    ) -> Callable:
        """装饰器：将函数注册为工具。

        Args:
            name: 工具名称，默认使用函数名。
            description: 工具描述，若不提供则使用函数 docstring 第一行。
            parameters: 参数列表，若不提供则根据函数签名自动推断。

        Returns:
            原函数（不修改函数行为）。

        示例::

            @registry.register(description="获取天气")
            def get_weather(city: str) -> str:
                \"\"\"查询指定城市天气。\"\"\"
                return f"{city}: 晴，25°C"
        """

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or (func.__doc__ or "").strip().split("\n")[0]
            tool_params = parameters if parameters is not None else _infer_parameters(func)

            self._tools[tool_name] = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                parameters=tool_params,
                func=func,
            )
            logger.debug("注册工具: %s", tool_name)
            return func

        return decorator

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter],
        func: Callable,
    ) -> None:
        """显式注册工具（非装饰器方式）。

        Args:
            name: 工具名称（全局唯一）。
            description: 工具描述。
            parameters: 参数列表。
            func: 执行函数。
        """
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )
        logger.debug("注册工具: %s", name)

    def unregister(self, name: str) -> bool:
        """注销工具。

        Returns:
            True 表示注销成功，False 表示工具不存在。
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def enable(self, name: str) -> None:
        """启用工具（使其对 LLM 可见）。"""
        if name in self._tools:
            self._tools[name].enabled = True

    def disable(self, name: str) -> None:
        """禁用工具（对 LLM 不可见，但保留注册信息）。"""
        if name in self._tools:
            self._tools[name].enabled = False

    def get(self, name: str) -> ToolDefinition | None:
        """按名称查找工具定义。"""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """列出所有已注册工具的名称（含禁用工具）。"""
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """获取所有启用工具的 Anthropic 格式 Schema。

        此方法返回值可直接作为 ``tools`` 参数传给 MiniMaxClient.chat()。

        Returns:
            Anthropic 格式的工具定义列表。
        """
        return [
            tool.to_anthropic_schema()
            for tool in self._tools.values()
            if tool.enabled
        ]

    def execute(self, name: str, params: dict[str, Any]) -> Any:
        """执行指定工具。

        Args:
            name: 工具名称。
            params: 工具参数字典，键名与函数参数名一致。

        Returns:
            工具执行结果（任意类型）。

        Raises:
            KeyError: 工具未注册。
            Exception: 工具函数执行失败时透传原始异常。
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self.list_tools()) or "（无）"
            raise KeyError(f"未找到工具: '{name}'。已注册工具: [{available}]")
        return tool.execute(**params)

    def __len__(self) -> int:
        """返回已注册工具数量（含禁用工具）。"""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """检查工具是否已注册。"""
        return name in self._tools

    def __repr__(self) -> str:
        tools = list(self._tools.keys())
        return f"ToolRegistry(tools={tools})"


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _python_type_to_json_type(annotation: Any) -> str:
    """将 Python 类型注解转换为 JSON Schema 基础类型。"""
    origin = getattr(annotation, "__origin__", None)

    if annotation in (int, float):
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation in (str, type(None)):
        return "string"
    if origin in (list, tuple):
        return "array"
    if origin is dict:
        return "object"
    return "string"  # 默认 string，足够通用


def _infer_parameters(func: Callable) -> list[ToolParameter]:
    """根据函数签名自动推断工具参数列表。

    推断规则：
    - 参数名取自函数签名
    - 类型从 type hints 转换为 JSON Schema 类型
    - 有默认值的参数标记为非必填
    - 描述从 docstring Args 段落中提取（如有）
    """
    sig = inspect.signature(func)

    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    # 解析 docstring 中 "Args:" 段落的参数描述
    doc = func.__doc__ or ""
    param_docs: dict[str, str] = {}
    in_args_section = False

    for line in doc.split("\n"):
        stripped = line.strip()
        if stripped in ("Args:", "Arguments:", "参数:"):
            in_args_section = True
            continue
        if in_args_section:
            if stripped and not stripped[0].isspace() and ":" in stripped:
                # 非缩进行，Args 段落结束
                in_args_section = False
                continue
            # 匹配 "param_name: description" 格式
            for p_name in sig.parameters:
                if stripped.startswith(f"{p_name}:") or stripped.startswith(f"{p_name} ("):
                    desc = stripped.split(":", 1)[-1].strip()
                    if desc:
                        param_docs[p_name] = desc

    params: list[ToolParameter] = []
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        annotation = hints.get(param_name, str)
        json_type = _python_type_to_json_type(annotation)
        description = param_docs.get(param_name, f"参数 {param_name}")
        required = param.default is inspect.Parameter.empty

        params.append(
            ToolParameter(
                name=param_name,
                description=description,
                type=json_type,
                required=required,
            )
        )

    return params


# 全局默认注册表（可在任意模块中导入使用）
default_registry = ToolRegistry()
