"""
Context output formatter.

Provides convenience functions to format a :class:`ContextPack` for
different LLM consumption patterns:
    - Compact JSON (default)
    - Markdown
    - System-prompt injection
    - Chat-message attachment
"""

from __future__ import annotations

from irpai.context.schema import ContextPack


def format_for_llm(
    pack: ContextPack,
    *,
    style: str = "json",
    max_chars: int | None = None,
) -> str:
    """
    Format a context pack for LLM prompt injection.

    Parameters
    ----------
    pack:
        The assembled context pack.
    style:
        ``'json'`` (compact), ``'markdown'``, or ``'system_prompt'``.
    max_chars:
        Optional hard character limit.
    """
    if style == "markdown":
        text = pack.to_markdown()
    elif style == "system_prompt":
        text = _as_system_prompt(pack)
    else:
        text = pack.to_json(compact=True)

    if max_chars and len(text) > max_chars:
        text = text[:max_chars - 20] + "\n[context truncated]"

    return text


def _as_system_prompt(pack: ContextPack) -> str:
    """Wrap the context pack in a system-prompt–friendly format."""
    parts = [
        "<screenshot_context>",
        pack.to_json(compact=False, include_meta=False),
        "</screenshot_context>",
    ]
    if pack.summary.description:
        parts.insert(1, f"<!-- {pack.summary.description} -->")
    return "\n".join(parts)
