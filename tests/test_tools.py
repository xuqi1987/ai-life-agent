"""工具系统测试 — 测试 ToolRegistry 和内置工具的功能。"""

from __future__ import annotations

import pytest

from ai_life_agent.tools.builtin import (
    _calculator,
    _echo,
    _get_current_time,
    register_builtin_tools,
)
from ai_life_agent.tools.registry import ToolParameter, ToolRegistry, _infer_parameters

# ---------------------------------------------------------------------------
# ToolRegistry 基础功能测试
# ---------------------------------------------------------------------------


class TestToolRegistry:
    """测试 ToolRegistry 的注册、查找、执行功能。"""

    def test_register_via_decorator(self):
        """装饰器方式注册工具。"""
        registry = ToolRegistry()

        @registry.register(description="两数相加")
        def add(a: int, b: int) -> int:
            return a + b

        assert "add" in registry
        assert len(registry) == 1

    def test_register_explicit(self):
        """显式方式注册工具。"""
        registry = ToolRegistry()
        registry.register_tool(
            name="echo",
            description="原样返回",
            parameters=[ToolParameter("text", "输入文本", "string")],
            func=lambda text: text,
        )
        assert "echo" in registry

    def test_execute_tool(self):
        """执行已注册工具。"""
        registry = ToolRegistry()

        @registry.register(description="乘法")
        def multiply(x: int, y: int) -> int:
            return x * y

        result = registry.execute("multiply", {"x": 3, "y": 4})
        assert result == 12

    def test_execute_unknown_tool_raises(self):
        """执行未注册工具应抛出 KeyError。"""
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="未找到工具"):
            registry.execute("nonexistent", {})

    def test_unregister(self):
        """注销工具。"""
        registry = ToolRegistry()

        @registry.register()
        def my_tool() -> str:
            return "ok"

        assert "my_tool" in registry
        result = registry.unregister("my_tool")
        assert result is True
        assert "my_tool" not in registry

    def test_unregister_nonexistent(self):
        """注销不存在的工具应返回 False。"""
        registry = ToolRegistry()
        assert registry.unregister("ghost") is False

    def test_enable_disable(self):
        """启用/禁用工具。"""
        registry = ToolRegistry()

        @registry.register(description="测试工具")
        def test_fn() -> str:
            return "ok"

        # 禁用后不出现在 schemas 中
        registry.disable("test_fn")
        schemas = registry.get_schemas()
        assert not any(s["name"] == "test_fn" for s in schemas)

        # 启用后重新出现
        registry.enable("test_fn")
        schemas = registry.get_schemas()
        assert any(s["name"] == "test_fn" for s in schemas)

    def test_disabled_tool_raises_on_execute(self):
        """禁用的工具执行时应抛出 RuntimeError。"""
        registry = ToolRegistry()

        @registry.register(description="临时工具")
        def temp() -> str:
            return "ok"

        registry.disable("temp")
        with pytest.raises(RuntimeError, match="已被禁用"):
            registry.execute("temp", {})

    def test_list_tools(self):
        """列出所有工具。"""
        registry = ToolRegistry()
        registry.register_tool("t1", "T1", [], lambda: "t1")
        registry.register_tool("t2", "T2", [], lambda: "t2")
        tools = registry.list_tools()
        assert "t1" in tools
        assert "t2" in tools
        assert len(tools) == 2


# ---------------------------------------------------------------------------
# Anthropic Schema 生成测试
# ---------------------------------------------------------------------------


class TestSchemaGeneration:
    """测试工具 Schema 生成（Anthropic 格式）。"""

    def test_schema_structure(self):
        """Schema 结构符合 Anthropic 格式。"""
        registry = ToolRegistry()
        registry.register_tool(
            name="greet",
            description="打招呼",
            parameters=[
                ToolParameter("name", "名字", "string", required=True),
                ToolParameter("lang", "语言", "string", required=False, enum=["zh", "en"]),
            ],
            func=lambda name, lang="zh": f"你好 {name}",
        )

        schemas = registry.get_schemas()
        assert len(schemas) == 1

        schema = schemas[0]
        assert schema["name"] == "greet"
        assert schema["description"] == "打招呼"
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"

        props = schema["input_schema"]["properties"]
        assert "name" in props
        assert props["name"]["type"] == "string"
        assert "lang" in props
        assert props["lang"]["enum"] == ["zh", "en"]

        required = schema["input_schema"]["required"]
        assert "name" in required
        assert "lang" not in required

    def test_empty_registry_returns_empty_schemas(self):
        """空注册表返回空 Schema 列表。"""
        registry = ToolRegistry()
        assert registry.get_schemas() == []


# ---------------------------------------------------------------------------
# 参数自动推断测试
# ---------------------------------------------------------------------------


class TestParameterInference:
    """测试从函数签名自动推断参数。"""

    def test_infer_basic_types(self):
        """自动推断基本类型。"""

        def my_func(name: str, count: int, ratio: float, active: bool) -> str:
            return ""

        params = _infer_parameters(my_func)
        param_map = {p.name: p for p in params}

        assert param_map["name"].type == "string"
        assert param_map["count"].type == "number"
        assert param_map["ratio"].type == "number"
        assert param_map["active"].type == "boolean"

    def test_infer_required_vs_optional(self):
        """有默认值的参数标记为非必填。"""

        def my_func(required_param: str, optional_param: str = "default") -> str:
            return ""

        params = _infer_parameters(my_func)
        param_map = {p.name: p for p in params}

        assert param_map["required_param"].required is True
        assert param_map["optional_param"].required is False

    def test_skip_self_parameter(self):
        """跳过 self 参数。"""

        class MyClass:
            def method(self, text: str) -> str:
                return text

        params = _infer_parameters(MyClass.method)
        names = [p.name for p in params]
        assert "self" not in names
        assert "text" in names


# ---------------------------------------------------------------------------
# 内置工具测试
# ---------------------------------------------------------------------------


class TestBuiltinTools:
    """测试内置工具的功能。"""

    def setup_method(self):
        """每个测试前创建新注册表并注册内置工具。"""
        self.registry = ToolRegistry()
        register_builtin_tools(self.registry)

    def test_all_builtin_tools_registered(self):
        """所有内置工具都已注册。"""
        tools = self.registry.list_tools()
        assert "calculator" in tools
        assert "get_current_time" in tools
        assert "echo" in tools

    def test_calculator_basic_arithmetic(self):
        """计算器基本运算。"""
        assert _calculator("2 + 3") == "5"
        assert _calculator("10 - 4") == "6"
        assert _calculator("3 * 4") == "12"
        assert _calculator("10 / 4") == "2.5"

    def test_calculator_power(self):
        """计算器幂次运算。"""
        assert _calculator("2 ** 8") == "256"

    def test_calculator_math_functions(self):
        """计算器数学函数。"""
        assert _calculator("sqrt(16)") == "4"
        assert _calculator("abs(-5)") == "5"

    def test_calculator_pi(self):
        """计算器使用 pi 常量。"""
        result = _calculator("round(pi, 4)")
        assert result == "3.1416"

    def test_calculator_division_by_zero(self):
        """除数为零时返回错误信息。"""
        result = _calculator("1 / 0")
        assert "除数不能为零" in result

    def test_calculator_injection_prevention(self):
        """阻止代码注入攻击。"""
        # 尝试使用双下划线
        result = _calculator("__import__('os')")
        assert "不允许" in result

    def test_calculator_via_registry(self):
        """通过注册表调用计算器。"""
        result = self.registry.execute("calculator", {"expression": "100 / 4"})
        assert result == "25"

    def test_get_current_time_format(self):
        """获取当前时间，格式符合预期。"""
        result = _get_current_time()
        # 默认格式: YYYY-MM-DD HH:MM:SS
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result)

    def test_get_current_time_custom_format(self):
        """自定义时间格式。"""
        result = _get_current_time("%Y/%m/%d")
        import re
        assert re.match(r"\d{4}/\d{2}/\d{2}", result)

    def test_get_current_time_via_registry(self):
        """通过注册表调用 get_current_time，不传参数时使用默认值。"""
        result = self.registry.execute("get_current_time", {})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_echo(self):
        """echo 工具原样返回文本。"""
        assert _echo("Hello, World!") == "Hello, World!"
        assert _echo("") == ""
        assert _echo("你好，世界！") == "你好，世界！"

    def test_echo_via_registry(self):
        """通过注册表调用 echo。"""
        result = self.registry.execute("echo", {"text": "test"})
        assert result == "test"
