"""OpenAI-compatible client shim for Moonshot AI's Kimi models.

The agents were originally written against Google's ``google.genai`` SDK. This
module exposes the tiny subset of that interface the agents actually rely on
(``Client``, ``types.GenerateContentConfig``, ``models.generate_content`` and a
response object with a ``.text`` attribute) but routes every call to Moonshot
AI's OpenAI-compatible Chat Completions API. That lets the existing call sites
stay almost unchanged while talking to Kimi K2.6.

Import it in place of ``genai``::

    from . import llm_client as genai

Get an API key at https://platform.moonshot.ai and set it via ``config.yaml``
(``moonshot_api_key``) or the ``MOONSHOT_API_KEY`` environment variable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

# Moonshot AI's OpenAI-compatible endpoint. Override with MOONSHOT_BASE_URL if
# you route Kimi through a different provider (OpenRouter, Together, etc.).
DEFAULT_BASE_URL = os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")


@dataclass
class GenerateContentConfig:
    """Mirror of ``google.genai`` ``GenerateContentConfig`` for the fields used here."""

    system_instruction: str = ""
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None


class _Types:
    GenerateContentConfig = GenerateContentConfig


# Exposed so call sites can keep using ``genai.types.GenerateContentConfig(...)``.
types = _Types()


@dataclass
class _Response:
    """Stand-in for the genai response object; only ``.text`` is consumed."""

    text: str


class _Models:
    def __init__(self, client: OpenAI):
        self._client = client

    def generate_content(self, model, contents, config=None):
        messages = []
        system = getattr(config, "system_instruction", "") if config else ""
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": contents})

        kwargs = {}
        if config is not None:
            if config.max_output_tokens is not None:
                kwargs["max_tokens"] = config.max_output_tokens
            if config.temperature is not None:
                kwargs["temperature"] = config.temperature

        completion = self._client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )

        text = ""
        if completion.choices:
            text = completion.choices[0].message.content or ""
        return _Response(text=text)


class Client:
    """Drop-in stand-in for ``google.genai.Client`` backed by Moonshot's API."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.models = _Models(self._client)
