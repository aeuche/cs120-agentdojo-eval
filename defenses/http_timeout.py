"""Bound OpenAI client calls so a shard cannot sit on one socket forever.

``scripts/secondary_runs.py`` loads this before AgentDojo builds a pipeline.
The default client has no deadline, and a stuck completion kept three
``ignore_previous`` shards asleep for hours with no log output.
"""

from __future__ import annotations

import openai

_OPENAI = openai.OpenAI


def _openai_with_timeout(*args, **kwargs):
    kwargs.setdefault("timeout", 120.0)
    kwargs.setdefault("max_retries", 2)
    return _OPENAI(*args, **kwargs)


openai.OpenAI = _openai_with_timeout
