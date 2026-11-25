"""Apply the two study configuration changes before AgentDojo builds a pipeline.

``transformers_pi_detector`` is ProtectAI's DeBERTa detector. AgentDojo 0.1.35
constructs it with ``raise_on_injection`` false and threshold 0.5, so a
detection redacts the tool output and the agent continues. The proposal says
the detector blocks the run. This module sets ``raise_on_injection`` to true
after the library builds the element. The threshold stays 0.5.

``tool_filter`` raises ``ValueError`` unless the pipeline model is an
``OpenAILLM``. ``LOCAL`` is a ``LocalLLM``. For that provider this module
builds the same pipeline shape with ``LocalToolFilter``, which asks the served
model which tools the user task needs. OpenAI models keep the shipped filter.

Importing this module also records OpenAI chat-completion usage onto the
active trace as ``api_usage`` and appends ``runs/usage.jsonl``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

# Weights downloaded for the smoke live in the project cache, which is gitignored.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "huggingface"))

from agentdojo.agent_pipeline.agent_pipeline import (
    TOOL_FILTER_PROMPT,
    AgentPipeline,
    PipelineConfig,
    get_llm,
)
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.llms.local_llm import LocalLLM
from agentdojo.agent_pipeline.pi_detector import PromptInjectionDetector, TransformersBasedPIDetector
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str
from agentdojo.logging import LOGGER_STACK, TraceLogger
from agentdojo.models import MODEL_PROVIDERS, ModelsEnum

from defenses.local_tool_filter import LocalToolFilter

USAGE_LOG = Path("runs/usage.jsonl")
DETECTOR_THRESHOLD = 0.5

_APPLIED = False
_ORIGINAL_FROM_CONFIG = AgentPipeline.from_config.__func__
_ORIGINAL_TRACE_EXIT = TraceLogger.__exit__


def _walk(element):
    yield element
    children = getattr(element, "elements", None)
    if not children:
        return
    for child in children:
        yield from _walk(child)


def _usage_counts(usage) -> tuple[int, int, int] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", 0)
    else:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    if prompt is None or completion is None:
        return None
    return int(prompt), int(cached or 0), int(completion)


def _current_trace() -> TraceLogger | None:
    for logger in reversed(LOGGER_STACK.get()):
        if isinstance(logger, TraceLogger):
            return logger
    return None


def _record_usage(model: str | None, response) -> None:
    counts = _usage_counts(getattr(response, "usage", None))
    if counts is None:
        return
    prompt, cached, completion = counts
    event = {
        "model": str(model or getattr(response, "model", "")),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "completion_tokens": completion,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    trace = _current_trace()
    if trace is not None:
        calls = getattr(trace, "_cs120_calls", None)
        if calls is None:
            calls = []
            trace._cs120_calls = calls
        calls.append({key: event[key] for key in ("model", "prompt_tokens", "cached_tokens", "completion_tokens")})
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _install_usage_meter() -> None:
    from openai.resources.chat.completions.completions import Completions

    if getattr(Completions.create, "_cs120_meter", False):
        return
    original = Completions.create

    def create(self, *args, **kwargs):
        response = original(self, *args, **kwargs)
        _record_usage(kwargs.get("model"), response)
        return response

    create._cs120_meter = True  # type: ignore[attr-defined]
    Completions.create = create  # type: ignore[method-assign]


def _install_trace_usage() -> None:
    if getattr(TraceLogger.__exit__, "_cs120_usage", False):
        return

    def __exit__(self, exc_type, exc, tb):
        calls = list(getattr(self, "_cs120_calls", []))
        self.context["api_usage"] = {"calls": calls}
        return _ORIGINAL_TRACE_EXIT(self, exc_type, exc, tb)

    __exit__._cs120_usage = True  # type: ignore[attr-defined]
    TraceLogger.__exit__ = __exit__  # type: ignore[method-assign]


def _formatter(config: PipelineConfig):
    if config.tool_output_format == "json":
        return lambda result: tool_result_to_str(result, dump_fn=json.dumps)
    return tool_result_to_str


def _is_local_tool_filter(config: PipelineConfig) -> bool:
    if config.defense != "tool_filter":
        return False
    if isinstance(config.llm, LocalLLM):
        return True
    if not isinstance(config.llm, str):
        return False
    try:
        provider = MODEL_PROVIDERS[ModelsEnum(config.llm)]
    except ValueError:
        return False
    return provider == "local"


def _local_tool_filter_pipeline(config: PipelineConfig) -> AgentPipeline:
    llm = get_llm(MODEL_PROVIDERS[ModelsEnum(config.llm)], str(config.llm), config.model_id, config.tool_delimiter)
    if not isinstance(llm, LocalLLM):
        raise ValueError("Local tool filter expected a LocalLLM pipeline model")
    llm_name = str(config.llm)
    assert config.system_message is not None
    tools_loop = ToolsExecutionLoop([ToolsExecutor(_formatter(config)), llm])
    pipeline = AgentPipeline(
        [
            SystemMessage(config.system_message),
            InitQuery(),
            LocalToolFilter(TOOL_FILTER_PROMPT, llm.client, llm.model),
            llm,
            tools_loop,
        ]
    )
    pipeline.name = f"{llm_name}-tool_filter"
    logging.info(
        "tool_filter: LOCAL uses LocalToolFilter on model %s. Stock 0.1.35 refuses a non-OpenAI client.",
        llm.model,
    )
    return pipeline


def _enable_detector_abort(pipeline: AgentPipeline) -> None:
    detectors = [element for element in _walk(pipeline) if isinstance(element, TransformersBasedPIDetector)]
    if not detectors:
        raise RuntimeError("transformers_pi_detector pipeline has no TransformersBasedPIDetector element")
    for detector in detectors:
        detector.raise_on_injection = True
    threshold = detectors[0].threshold
    logging.info(
        "transformers_pi_detector: raise_on_injection=True, so a detection aborts the run. threshold=%s.",
        threshold,
    )


def _from_config(cls, config: PipelineConfig):
    if _is_local_tool_filter(config):
        return _local_tool_filter_pipeline(config)
    pipeline = _ORIGINAL_FROM_CONFIG(cls, config)
    if config.defense == "transformers_pi_detector":
        _enable_detector_abort(pipeline)
    return pipeline


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    _install_usage_meter()
    _install_trace_usage()
    AgentPipeline.from_config = classmethod(_from_config)  # type: ignore[method-assign]


def _find(pipeline, kind):
    return [element for element in _walk(pipeline) if isinstance(element, kind)]


def self_check() -> None:
    """Check the abort flag, the local filter, and that the stock filter still refuses LOCAL."""
    import os

    os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")
    apply()
    local_config = PipelineConfig(
        llm="local",
        model_id="llama-3.1-8b-instruct-q",
        defense="tool_filter",
        system_message_name=None,
        system_message=None,
    )
    try:
        _ORIGINAL_FROM_CONFIG(AgentPipeline, local_config)
    except ValueError as exc:
        if "OpenAI" not in str(exc):
            raise AssertionError(exc) from exc
    else:
        raise AssertionError("stock tool_filter accepted LOCAL")

    local_pipeline = AgentPipeline.from_config(local_config)
    if not str(local_pipeline.name).endswith("-tool_filter"):
        raise AssertionError(local_pipeline.name)
    filters = _find(local_pipeline, LocalToolFilter)
    if len(filters) != 1 or filters[0].prompt != TOOL_FILTER_PROMPT or filters[0].temperature != 0.0:
        raise AssertionError("LocalToolFilter was not installed with the shipped prompt at temperature 0")

    class _Tool:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

    class _Runtime:
        def __init__(self) -> None:
            self.functions = {
                "search_emails": _Tool("search_emails", "Search the inbox"),
                "search_files_by_filename": _Tool("search_files_by_filename", "Find a file by name"),
                "delete_file": _Tool("delete_file", "Delete a file"),
            }

        def update_functions(self, new) -> None:
            self.functions = new

    class _Message:
        content = "search_emails, search_files"

    class _Choice:
        message = _Message()

    class _Completion:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            _Completions.kwargs = kwargs
            return _Completion()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    runtime = _Runtime()
    element = LocalToolFilter(TOOL_FILTER_PROMPT, _Client(), "local-id")
    _, runtime, _, messages, _ = element.query("Read my mail", runtime, messages=[])
    kept = set(runtime.functions)
    if kept != {"search_emails"}:
        raise AssertionError(kept)
    if _Completions.kwargs["temperature"] != 0.0 or _Completions.kwargs["model"] != "local-id":
        raise AssertionError(_Completions.kwargs)
    if len(messages) != 2:
        raise AssertionError(len(messages))

    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLMToolFilter

    openai_filter = AgentPipeline.from_config(
        PipelineConfig(
            llm="gpt-4o-mini-2024-07-18",
            model_id=None,
            defense="tool_filter",
            system_message_name=None,
            system_message=None,
        )
    )
    if len(_find(openai_filter, OpenAILLMToolFilter)) != 1 or _find(openai_filter, LocalToolFilter):
        raise AssertionError("OpenAI tool_filter should keep the shipped filter")

    original_init = TransformersBasedPIDetector.__init__

    def stub_init(
        self,
        model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
        safe_label: str = "SAFE",
        threshold: float = 0.5,
        mode: str = "message",
        raise_on_injection: bool = False,
    ) -> None:
        PromptInjectionDetector.__init__(self, mode=mode, raise_on_injection=raise_on_injection)
        self.model_name = model_name
        self.safe_label = safe_label
        self.threshold = threshold

    TransformersBasedPIDetector.__init__ = stub_init  # type: ignore[method-assign]
    try:
        detector_pipeline = AgentPipeline.from_config(
            PipelineConfig(
                llm="gpt-4o-mini-2024-07-18",
                model_id=None,
                defense="transformers_pi_detector",
                system_message_name=None,
                system_message=None,
            )
        )
    finally:
        TransformersBasedPIDetector.__init__ = original_init  # type: ignore[method-assign]
    detectors = _find(detector_pipeline, TransformersBasedPIDetector)
    if len(detectors) != 1:
        raise AssertionError(detectors)
    if detectors[0].raise_on_injection is not True:
        raise AssertionError("detector still redacts instead of aborting")
    if detectors[0].threshold != DETECTOR_THRESHOLD:
        raise AssertionError(detectors[0].threshold)
    if not str(detector_pipeline.name).endswith("-transformers_pi_detector"):
        raise AssertionError(detector_pipeline.name)
    print("defenses.study self-check passed")


apply()


if __name__ == "__main__":
    self_check()
