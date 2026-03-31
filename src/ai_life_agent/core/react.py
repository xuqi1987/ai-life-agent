"""ReAct 推理循环 — Reasoning + Acting 的核心实现。

ReAct（Reasoning + Acting）是一种让 LLM 交替进行推理（Thought）和行动（Action）的范式。
通过迭代循环，Agent 可以自主决定下一步操作，并根据工具执行结果调整策略。

原论文: https://arxiv.org/abs/2210.03629

循环流程示例::

    用户: "2+3*4 等于多少？顺便告诉我现在几点？"

    [迭代 1]
    Thought: 我需要先计算 2+3*4
    Action: calculator(expression="2+3*4")
    Observation: 14

    [迭代 2]
    Thought: 计算完成，现在查询时间
    Action: get_current_time()
    Observation: 2026-03-31 14:00:00

    [迭代 3 — 最终回答]
    Answer: 2+3*4 = 14。现在时间是 2026-03-31 14:00:00。

实现细节：
- 使用 Anthropic SDK 工具调用格式（tool_use / tool_result）
- 多轮对话中，将完整的 assistant 消息（含 thinking block）回传，保持思维链连续性
- 最大迭代次数限制防止无限循环
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ai_life_agent.config import settings
from ai_life_agent.llm.client import MiniMaxClient
from ai_life_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# ReAct 系统提示词 — 引导模型进行 Thought→Action→Observation 推理
REACT_SYSTEM_PROMPT = """你是一个智能 AI 助手，名字叫 AI Life Agent，由 MiniMax M2.7 驱动。

你有访问各种工具的能力。在回答用户问题时，请遵循以下原则：

1. **先思考，再行动**：在调用工具前，先分析用户意图，确定需要哪些信息
2. **善用工具**：如果问题需要实时信息（时间、天气等）或计算，请使用对应工具
3. **基于事实回答**：将工具返回的结果作为事实依据，组织清晰的回答
4. **中文优先**：默认使用中文回答，除非用户明确使用其他语言

你拥有的工具能力：
- 数学计算（calculator）
- 获取当前时间（get_current_time）
- 文字转语音（tts_speak，如果已配置）
- 其他扩展工具

请专注于理解并解决用户的问题，给出有帮助的回答。"""


@dataclass
class ReActStep:
    """单次 ReAct 迭代的记录。

    Attributes:
        iteration: 迭代序号（从 1 开始）。
        thinking: 模型的思维链内容（如有）。
        tool_calls: 本轮触发的工具调用列表。
        observations: 工具执行结果列表。
        final_answer: 若本轮为最终答案，此字段非空。
    """

    iteration: int
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    final_answer: str | None = None

    def is_final(self) -> bool:
        """是否为最终答案步骤。"""
        return self.final_answer is not None


@dataclass
class ReActResult:
    """ReAct 循环的最终结果。

    Attributes:
        answer: 最终回答文本。
        steps: 完整的执行步骤列表（可用于调试和展示）。
        total_iterations: 总迭代次数。
        stopped_by_limit: 是否因达到最大迭代次数而强制停止。
    """

    answer: str
    steps: list[ReActStep] = field(default_factory=list)
    total_iterations: int = 0
    stopped_by_limit: bool = False

    def format_trace(self) -> str:
        """格式化完整的推理链路（用于调试输出）。"""
        lines = []
        for step in self.steps:
            lines.append(f"\n[迭代 {step.iteration}]")
            if step.thinking:
                # 只显示思维链前 200 字符
                thinking_preview = step.thinking[:200] + ("..." if len(step.thinking) > 200 else "")
                lines.append(f"  思维链: {thinking_preview}")
            for tc in step.tool_calls:
                lines.append(f"  调用工具: {tc['name']}({json.dumps(tc['input'], ensure_ascii=False)})")
            for obs in step.observations:
                lines.append(f"  工具结果: {obs}")
            if step.final_answer:
                lines.append(f"  最终回答: {step.final_answer[:100]}...")

        if self.stopped_by_limit:
            lines.append(f"\n⚠️  已达到最大迭代次数限制 ({self.total_iterations})")

        return "\n".join(lines)


class ReActExecutor:
    """ReAct 执行器 — 实现 Thought→Action→Observation 迭代循环。

    通过与 MiniMax M2.7 的多轮对话，自主决策并调用工具，直到给出最终答案。

    示例::

        from ai_life_agent.core.react import ReActExecutor
        from ai_life_agent.tools.registry import ToolRegistry
        from ai_life_agent.tools.builtin import register_builtin_tools

        registry = ToolRegistry()
        register_builtin_tools(registry)

        executor = ReActExecutor(registry=registry)
        result = executor.run("现在几点了？顺便帮我算一下 100 的平方根。")

        print(result.answer)
        print(result.format_trace())  # 查看完整推理链路
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        llm_client: MiniMaxClient | None = None,
        system_prompt: str | None = None,
        max_iterations: int | None = None,
        verbose: bool | None = None,
    ):
        """初始化 ReAct 执行器。

        Args:
            registry: 工具注册表，若为 None 则使用空注册表（无工具）。
            llm_client: LLM 客户端，若为 None 则自动创建（使用 settings 配置）。
            system_prompt: 系统提示词，若为 None 则使用默认 REACT_SYSTEM_PROMPT。
            max_iterations: 最大迭代次数，若为 None 则使用配置值。
            verbose: 是否输出详细日志，若为 None 则使用配置值。
        """
        self.registry = registry or ToolRegistry()
        self.llm = llm_client or MiniMaxClient()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.max_iterations = max_iterations or settings.max_react_iterations
        self.verbose = verbose if verbose is not None else settings.verbose

    def run(
        self,
        user_input: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> ReActResult:
        """执行 ReAct 循环，处理用户输入并返回最终答案。

        Args:
            user_input: 用户输入文本。
            conversation_history: 历史对话（用于多轮对话上下文），格式为::

                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Returns:
            ReActResult 对象，包含最终答案和完整推理链路。
        """
        # 构建初始消息列表
        messages: list[dict[str, Any]] = list(conversation_history or [])
        messages.append({"role": "user", "content": user_input})

        # 获取当前可用的工具 Schema
        tools = self.registry.get_schemas()

        steps: list[ReActStep] = []
        stopped_by_limit = False

        for iteration in range(1, self.max_iterations + 1):
            step = ReActStep(iteration=iteration)

            if self.verbose:
                logger.info("[ReAct 迭代 %d/%d] 发送消息给 LLM...", iteration, self.max_iterations)

            # 调用 LLM
            try:
                response = self.llm.chat(
                    messages=messages,
                    system=self.system_prompt,
                    tools=tools if tools else None,
                )
            except Exception as e:
                logger.error("LLM 调用失败: %s", e)
                step.final_answer = f"[LLM 调用失败: {e}]"
                steps.append(step)
                break

            step.thinking = response.thinking

            if self.verbose and response.thinking:
                logger.info("  思维链: %s...", response.thinking[:150])

            # Case 1: 无工具调用 → 最终答案
            if not response.has_tool_calls:
                step.final_answer = response.text
                steps.append(step)

                if self.verbose:
                    logger.info("  最终回答: %s...", response.text[:100])
                break

            # Case 2: 有工具调用 → 执行工具 → 继续迭代
            step.tool_calls = response.tool_calls

            # 将 assistant 完整消息（含 thinking block）加入历史
            # 注意：MiniMax 要求多轮工具调用时必须回传完整的 content 列表
            messages.append(
                {"role": "assistant", "content": response.raw_content}
            )

            # 执行所有工具调用，收集结果
            tool_results: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_input = tc["input"]
                tool_call_id = tc["id"]

                if self.verbose:
                    logger.info(
                        "  执行工具: %s(%s)",
                        tool_name,
                        json.dumps(tool_input, ensure_ascii=False),
                    )

                try:
                    observation = self.registry.execute(tool_name, tool_input)
                    # 将结果统一转为字符串
                    observation_str = (
                        observation if isinstance(observation, str) else str(observation)
                    )
                except Exception as e:
                    observation_str = f"工具执行失败: {e}"
                    logger.warning("工具 %s 执行失败: %s", tool_name, e)

                if self.verbose:
                    logger.info("  工具结果: %s", observation_str)

                step.observations.append(observation_str)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": observation_str,
                    }
                )

            # 将工具结果作为 user 消息回传
            messages.append({"role": "user", "content": tool_results})
            steps.append(step)

        else:
            # 超出最大迭代次数
            stopped_by_limit = True
            logger.warning("ReAct 循环达到最大迭代次数 (%d)，强制停止", self.max_iterations)
            # 取最后一个有文本的响应作为答案
            final_answer = "（已达到最大迭代次数，以下为最后一次响应）"
            for step in reversed(steps):
                if step.final_answer:
                    final_answer = step.final_answer
                    break

        # 确定最终答案
        final_answer = ""
        for step in steps:
            if step.final_answer is not None:
                final_answer = step.final_answer
                break

        if not final_answer:
            final_answer = "（Agent 未能生成回答）"

        return ReActResult(
            answer=final_answer,
            steps=steps,
            total_iterations=len(steps),
            stopped_by_limit=stopped_by_limit,
        )
