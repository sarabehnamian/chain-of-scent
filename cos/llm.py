"""Thin LLM layer.

Two backends behind one interface so the whole pipeline can be developed and
debugged offline before spending a single token:

  AnthropicLLM  real calls, with tool use
  MockLLM       deterministic pseudo-agent that emits plausible reasoning and
                proposals by mutating molecules from a seed list

Set COS_LLM=mock (or pass backend="mock") to run everything offline.

Note on SDK versions: anthropic 1.0 (August 2026) removed temperature, top_p
and top_k from messages.create(); passing them raises TypeError. Older models
that still accept them need extra_body={"temperature": ...}. This module
handles both: temperature is sent only when explicitly set AND the SDK is
pre-1.0, otherwise it is dropped.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

DEFAULT_MODEL = "claude-sonnet-5"


@dataclass
class Turn:
    role: str
    content: object


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    raw: object = None
    stop_reason: str = ""
    usage: dict = field(default_factory=dict)


def _sdk_major() -> int:
    try:
        import anthropic

        return int(str(getattr(anthropic, "__version__", "0")).split(".")[0])
    except Exception:
        return 0


class AnthropicLLM:
    """Wrapper over the Messages API. Requires ANTHROPIC_API_KEY."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1200,
                 temperature: float | None = None, max_retries: int = 5):
        import anthropic  # lazy: not needed for mock runs

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self._sdk_major = _sdk_major()

    def __call__(self, system: str, messages: list[dict],
                 tools: list[dict] | None = None) -> LLMResponse:
        import anthropic

        kwargs: dict = dict(model=self.model, max_tokens=self.max_tokens,
                            system=system, messages=messages)
        if tools:
            kwargs["tools"] = tools
        if self.temperature is not None:
            if self._sdk_major >= 1:
                # removed from the typed signature in SDK 1.0; only older
                # models accept it at all, and then only via extra_body
                kwargs["extra_body"] = {"temperature": self.temperature}
            else:
                kwargs["temperature"] = self.temperature

        delay = 2.0
        resp = None
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(**kwargs)
                break
            except (anthropic.RateLimitError, anthropic.APIStatusError,
                    anthropic.APIConnectionError) as err:
                last_err = err
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        if resp is None:
            raise RuntimeError(f"no response after retries: {last_err}")

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            {"id": b.id, "name": b.name, "input": b.input}
            for b in resp.content if b.type == "tool_use"
        ]
        return LLMResponse(
            text=text, tool_calls=calls, raw=resp, stop_reason=resp.stop_reason,
            usage={"input": resp.usage.input_tokens,
                   "output": resp.usage.output_tokens},
        )


class MockLLM:
    """Offline stand-in. Produces a reasoning paragraph that names real
    functional groups and descriptors, then calls the tool with a mutated
    molecule. Good enough to exercise every downstream code path, including
    claim extraction and intervention, without an API key."""

    SEEDS = ["CC/C=C\\CC=O", "CCCCCC=O", "CC(C)=CCCC(C)=CCO", "COc1ccccc1C=O",
             "CCCCC(=O)OCC", "CC(=O)CCc1ccccc1", "O=C1CCCCO1", "CCSCC",
             "CC1=CCC(CC1)C(C)C", "CCCCCCCC=O"]
    TEMPLATES = [
        "The {group} should push this toward {desc}, since that group is the "
        "usual carrier of {desc} character in small molecules.",
        "I expect a {desc} note here because of the {group}; removing it would "
        "lose most of that character.",
        "To raise {desc} I am adding chain length and keeping the {group} intact, "
        "which drives the {desc} impression.",
    ]

    def __init__(self, seed: int = 0, model: str = "mock"):
        self.rng = np.random.default_rng(seed)
        self.model = model
        self.n_calls = 0

    def __call__(self, system: str, messages: list[dict],
                 tools: list[dict] | None = None) -> LLMResponse:
        from . import chem

        self.n_calls += 1
        target = re.findall(r"target odour profile:\s*(.+)", system, re.I)
        descs = [d.strip() for d in target[0].split(",")] if target else ["green"]

        prev = None
        for m in reversed(messages):
            blob = json.dumps(m.get("content", ""))
            hit = re.findall(r'"smiles":\s*"([^"]+)"', blob)
            if hit:
                prev = hit[-1]
                break
        parent = prev or str(self.rng.choice(self.SEEDS))
        child, _ = chem.random_mutate(parent, self.rng)
        child = child or parent

        groups = chem.groups_present(child) or ["alkene"]
        g = str(self.rng.choice(groups))
        d = str(self.rng.choice(descs))
        text = str(self.rng.choice(self.TEMPLATES)).format(group=g.replace("_", " "),
                                                           desc=d)
        if tools:
            return LLMResponse(
                text=text,
                tool_calls=[{"id": f"mock_{self.n_calls}", "name": "predict_odor",
                             "input": {"smiles": child}}],
                stop_reason="tool_use",
                usage={"input": 0, "output": 0},
            )
        return LLMResponse(text=text + f"\nFINAL: {child}", stop_reason="end_turn",
                           usage={"input": 0, "output": 0})


def get_llm(backend: str | None = None, **kw):
    backend = backend or os.environ.get("COS_LLM", "mock")
    if backend == "mock":
        return MockLLM(**{k: v for k, v in kw.items() if k in ("seed", "model")})
    return AnthropicLLM(**{k: v for k, v in kw.items()
                           if k in ("model", "max_tokens", "temperature")})
