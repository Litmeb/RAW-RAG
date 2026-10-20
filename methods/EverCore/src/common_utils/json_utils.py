"""
JSON parsing utilities

Provides robust JSON extraction from LLM responses that may contain
markdown code blocks or extra text around JSON objects.
"""

import json
from typing import Any, Dict, Optional


def extract_outermost_json(text: str) -> Optional[str]:
    """Extract the outermost JSON object from text using bracket counting.

    Handles escaped characters and strings correctly.

    Args:
        text: Raw text potentially containing a JSON object

    Returns:
        The JSON substring, or None if no valid object found
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def repair_invalid_json_escapes(text: str) -> str:
    """Repair stray backslashes inside JSON strings.

    LLM output occasionally emits Windows paths, regex snippets, or plain text
    like ``\\d`` inside JSON string values. Those are invalid JSON escapes and
    make ``json.loads`` fail with ``Invalid \\escape``. This helper preserves
    valid JSON escapes and only doubles suspicious backslashes that appear
    inside quoted strings.
    """
    result: list[str] = []
    in_string = False
    i = 0

    while i < len(text):
        ch = text[i]

        if ch == '"':
            backslash_count = 0
            j = i - 1
            while j >= 0 and text[j] == "\\":
                backslash_count += 1
                j -= 1
            if backslash_count % 2 == 0:
                in_string = not in_string
            result.append(ch)
            i += 1
            continue

        if ch == "\\" and in_string:
            if i + 1 >= len(text):
                result.append("\\\\")
                i += 1
                continue

            next_ch = text[i + 1]
            if next_ch in '"\\/bfnrt':
                result.append(ch)
                result.append(next_ch)
                i += 2
                continue

            if next_ch == "u" and i + 5 < len(text):
                hex_part = text[i + 2 : i + 6]
                if all(c in "0123456789abcdefABCDEF" for c in hex_part):
                    result.append(text[i : i + 6])
                    i += 6
                    continue

            result.append("\\\\")
            i += 1
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from an LLM response, tolerating markdown code blocks and surrounding text.

    Tries in order:
    1. Extract from ```json ... ``` code block
    2. Extract outermost { ... } via bracket counting
    3. Parse the entire response as JSON

    Args:
        response: Raw LLM response string

    Returns:
        Parsed dict, or None if parsing fails
    """
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end > start:
            json_candidate = response[start:end].strip()
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                try:
                    return json.loads(repair_invalid_json_escapes(json_candidate))
                except json.JSONDecodeError:
                    pass

    json_str = extract_outermost_json(response)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                return json.loads(repair_invalid_json_escapes(json_str))
            except json.JSONDecodeError:
                pass

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            return json.loads(repair_invalid_json_escapes(response))
        except json.JSONDecodeError:
            return None
