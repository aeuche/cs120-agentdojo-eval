"""Tool filter for a local OpenAI-compatible model.

AgentDojo 0.1.35 only builds ``tool_filter`` for ``OpenAILLM``. ``LOCAL`` is a
``LocalLLM``, and ``AgentPipeline.from_config`` raises ``ValueError``. This
element asks that same served model which tools the user task needs, using
AgentDojo's ``TOOL_FILTER_PROMPT``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)


def message_text(message: ChatMessage) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return get_text_content_as_str(content)


def completion_text(completion) -> str:
    content = completion.choices[0].message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            parts.append(str(part.get("text") or part.get("content") or ""))
        else:
            parts.append(str(getattr(part, "text", "") or ""))
    return "".join(parts)


def whole_name(name: str, text: str) -> bool:
    """True when ``name`` appears as its own token, not as part of a longer tool name."""
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text) is not None


class LocalToolFilter(BasePipelineElement):
    """Ask the served local model for the tools the user task needs, then drop the rest.

    The call uses the same client and model id as the agent. Temperature is 0.
    Tool names are kept when they appear as whole tokens in the reply. A reply
    that names nothing leaves the tool set empty, which is the shipped OpenAI
    filter's result when no name matches.
    """

    def __init__(self, prompt: str, client, model: str, temperature: float = 0.0) -> None:
        self.prompt = prompt
        self.client = client
        self.model = model
        self.temperature = temperature

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        catalog = "\n".join(f"{tool.name}: {tool.description}" for tool in runtime.functions.values())
        filter_text = f"{self.prompt}\n\nAvailable tools:\n{catalog}"
        filter_message = ChatUserMessage(role="user", content=[text_content_block_from_string(filter_text)])
        payload = []
        for message in (*messages, filter_message):
            role = message["role"] if message["role"] in {"system", "user", "assistant"} else "user"
            payload.append({"role": role, "content": message_text(message)})
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature,
        )
        text = completion_text(completion)
        kept = {name: tool for name, tool in runtime.functions.items() if whole_name(name, text)}
        runtime.update_functions(kept)
        assistant = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(text)],
            tool_calls=None,
        )
        return query, runtime, env, [*messages, filter_message, assistant], extra_args
