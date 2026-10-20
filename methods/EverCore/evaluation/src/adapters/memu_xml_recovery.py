"""Tolerant parsing for memU's LLM-generated memory XML.

memU 1.5.1 parses the complete response as strict XML.  One malformed tag
therefore drops every memory in that response.  This module keeps the strict
path for well-formed output and salvages individually recognizable memory
blocks when an LLM produces slightly malformed XML.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

_ROOT_TAGS = ("item", "profile", "behaviors", "events", "knowledge", "skills")
_BARE_AMPERSAND = re.compile(
    r"&(?!amp;|lt;|gt;|quot;|apos;|#[0-9]+;|#x[0-9a-fA-F]+;)"
)
_MEMORY_START = re.compile(r"<memory\b[^>]*>", re.IGNORECASE)
_MEMORY_END = re.compile(r"</memory\s*>", re.IGNORECASE)
_STRUCTURAL_TAG = re.compile(
    r"</?(?:item|profile|behaviors|events|knowledge|skills|memory|content|"
    r"categories|category)\b[^>]*>",
    re.IGNORECASE,
)


def _xml_fragment(raw: str) -> str | None:
    """Return the first supported root element, excluding surrounding prose."""
    candidates: list[tuple[int, str]] = []
    for tag in _ROOT_TAGS:
        match = re.search(rf"<{tag}\b[^>]*>", raw, re.IGNORECASE)
        if match:
            candidates.append((match.start(), tag))
    if not candidates:
        return None

    start, root_tag = min(candidates)
    closing_matches = list(
        re.finditer(rf"</{root_tag}\s*>", raw[start:], re.IGNORECASE)
    )
    if not closing_matches:
        return raw[start:]
    closing = closing_matches[-1]
    return raw[start : start + closing.end()]


def _escape_bare_ampersands(value: str) -> str:
    """Escape only invalid bare ampersands, preserving existing XML entities."""
    return _BARE_AMPERSAND.sub("&amp;", value)


def _element_text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _strict_parse(fragment: str) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(_escape_bare_ampersands(fragment))
    memories: list[dict[str, Any]] = []
    for memory_element in root.findall("memory"):
        content = _element_text(memory_element.find("content"))
        categories_element = memory_element.find("categories")
        categories = (
            [
                text
                for category in categories_element.findall("category")
                if (text := _element_text(category))
            ]
            if categories_element is not None
            else []
        )
        if content and categories:
            memories.append({"content": content, "categories": categories})
    return memories


def _extract_field(
    block: str,
    tag: str,
    *,
    stop_before: tuple[str, ...] = (),
    clean: bool = True,
) -> str:
    opening = re.search(rf"<{tag}\b[^>]*>", block, re.IGNORECASE)
    if not opening:
        return ""

    tail = block[opening.end() :]
    endpoints: list[int] = []
    closing = re.search(rf"</{tag}\s*>", tail, re.IGNORECASE)
    if closing:
        endpoints.append(closing.start())
    for stop_tag in stop_before:
        stop = re.search(rf"<{stop_tag}\b[^>]*>", tail, re.IGNORECASE)
        if stop:
            endpoints.append(stop.start())
    value = tail[: min(endpoints)] if endpoints else tail
    return _clean_recovered_text(value) if clean else value.strip()


def _clean_recovered_text(value: str) -> str:
    """Remove leaked schema tags while retaining ordinary angle-bracket text."""
    return html.unescape(_STRUCTURAL_TAG.sub("", value).strip())


def _recover_memory_block(block: str) -> dict[str, Any] | None:
    content = _extract_field(block, "content", stop_before=("categories",))
    categories_block = _extract_field(block, "categories", clean=False)
    categories: list[str] = []
    for opening in re.finditer(r"<category\b[^>]*>", categories_block, re.IGNORECASE):
        tail = categories_block[opening.end() :]
        endpoints: list[int] = []
        closing = re.search(r"</category\s*>", tail, re.IGNORECASE)
        if closing:
            endpoints.append(closing.start())
        next_category = re.search(r"<category\b[^>]*>", tail, re.IGNORECASE)
        if next_category:
            endpoints.append(next_category.start())
        value = tail[: min(endpoints)] if endpoints else tail
        value = _clean_recovered_text(value)
        if value:
            categories.append(value)

    if content and categories:
        return {"content": content, "categories": categories}
    return None


def _tolerant_parse(fragment: str) -> list[dict[str, Any]]:
    starts = list(_MEMORY_START.finditer(fragment))
    recovered: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        next_start = starts[index + 1].start() if index + 1 < len(starts) else len(fragment)
        end_match = _MEMORY_END.search(fragment, start.end(), next_start)
        end = end_match.start() if end_match else next_start
        if parsed := _recover_memory_block(fragment[start.end() : end]):
            recovered.append(parsed)
    return recovered


def parse_memory_type_response_xml(raw: str) -> list[dict[str, Any]]:
    """Parse memU memory XML and recover usable records from malformed output."""
    if not raw or not raw.strip():
        return []
    fragment = _xml_fragment(raw.strip())
    if fragment is None:
        logger.warning("memU XML response has no supported root tag")
        return []

    try:
        return _strict_parse(fragment)
    except ElementTree.ParseError as exc:
        recovered = _tolerant_parse(fragment)
        if recovered:
            logger.warning(
                "Recovered %d memU memory item(s) from malformed XML: %s",
                len(recovered),
                exc,
            )
        else:
            logger.warning("Could not recover malformed memU XML: %s", exc)
        return recovered
