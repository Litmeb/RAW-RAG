"""Full-history baseline with a budget for the complete chat request."""

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path

from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import SearchResult
from evaluation.src.utils.config import load_yaml


@register_adapter("full_context")
class FullContextAdapter(BaseAdapter):
    def __init__(self, config, output_dir=None):
        super().__init__(config)
        self.output_dir = output_dir
        self.llm = config.get("llm", {})
        self.options = config.get("answer", {})
        self.window = int(self.options.get("context_window", 128000))
        self.output_tokens = int(self.llm.get("max_tokens", 4096))
        self.margin = int(self.options.get("context_safety_margin", 256))
        self.strategy = self.options.get("truncation_strategy", "head_tail")
        if self.output_tokens <= 0 or self.margin < 0 or self.window <= self.output_tokens + self.margin:
            raise ValueError("context_window must exceed max_tokens + context_safety_margin")
        if self.strategy not in {"head", "tail", "head_tail"}:
            raise ValueError("truncation_strategy must be head, tail or head_tail")
        import tiktoken

        tokenizer = self.options.get("tokenizer_model") or self.llm.get("model", "gpt-4o-mini")
        try:
            self.encoding = tiktoken.encoding_for_model(tokenizer.removeprefix("openai/"))
        except KeyError:
            # Unknown providers: byte counting is deliberately conservative;
            # users can supply a matching tiktoken model/encoding explicitly.
            self.encoding = tiktoken.get_encoding(tokenizer) if tokenizer in tiktoken.list_encoding_names() else None
        self.prompts = load_yaml(str(Path(__file__).parents[2] / "config" / "prompts.yaml"))["online_api"]["default"]
        self._client = None

    async def add(self, conversations, **kwargs):
        return self.build_lazy_index(conversations, kwargs.get("output_dir"))

    def build_lazy_index(self, conversations, output_dir):
        # The dataset is already loaded on resume; no duplicate on-disk index.
        return {conv.conversation_id: conv for conv in conversations}

    @staticmethod
    def _date(value):
        if isinstance(value, datetime):
            parsed = value
        else:
            text = str(value).strip()
            parsed = None
            for fmt in ("%Y/%m/%d (%a) %H:%M", "%I:%M %p on %d %B, %Y"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    pass
            if parsed is None:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed

    async def search(self, query, conversation_id, index, **kwargs):
        conv = (index or {}).get(conversation_id) or kwargs.get("conversation")
        if conv is None or conv.conversation_id != conversation_id:
            raise ValueError(f"Missing conversation: {conversation_id}")
        cutoff = self._date(kwargs["question_date"]) if kwargs.get("question_date") else None
        lines = []
        excluded = 0
        for msg in conv.messages:
            if cutoff and (msg.timestamp is None or self._date(msg.timestamp) > cutoff):
                excluded += 1
                continue
            if self.config.get("dataset_name") == "memoryagentbench":
                lines.append(msg.content)
            else:
                stamp = f"[{msg.timestamp.isoformat()}] " if msg.timestamp else ""
                lines.append(f"{stamp}{msg.sender_name or msg.sender_id}: {msg.content}")
        return SearchResult(
            query=query, conversation_id=conversation_id, results=[],
            question_id=kwargs.get("question_id", ""),
            retrieval_metadata={"formatted_context": "\n\n".join(lines),
                                "visible_messages": len(lines), "excluded_messages": excluded,
                                "temporal_filter_enabled": cutoff is not None},
        )

    def _messages(self, query, context, prompt_key=None):
        if prompt_key and prompt_key.startswith("answer_prompt_mab_"):
            return [
                {"role": "system", "content": "You are a helpful AI. Answer the question based on query and memories.\n" + context + "\n"},
                {"role": "user", "content": query},
            ]
        key = prompt_key or self.options.get("prompt_key", "answer_prompt_cot")
        return [{"role": "user", "content": self.prompts[key].format(context=context, question=query)}]

    def _count(self, text):
        return len(self.encoding.encode(text, disallowed_special=())) if self.encoding else len(text.encode("utf-8"))

    def _request_tokens(self, messages):
        # Include role/framing overhead as well as all prompt text.
        return 3 + sum(8 + self._count(m["role"]) + self._count(m["content"]) for m in messages)

    def prepare_answer_context(self, query, context, prompt_key=None):
        limit = self.window - self.output_tokens - self.margin
        def count(candidate):
            return self._request_tokens(self._messages(query, candidate, prompt_key))
        if count("") > limit:
            raise ValueError("Question and prompt exceed context_window even without history")
        original_chars = len(context)
        if count(context) > limit:
            marker = "\n[... history truncated ...]\n"
            def candidate(size):
                if size == 0:
                    return ""
                if self.strategy == "head":
                    return context[:size] + marker
                if self.strategy == "tail":
                    return marker + context[-size:]
                head = (size + 1) // 2
                tail = size // 2
                return context[:head] + marker + (context[-tail:] if tail else "")
            low, high = 0, len(context)
            while low < high:
                mid = (low + high + 1) // 2
                if count(candidate(mid)) <= limit:
                    low = mid
                else:
                    high = mid - 1
            context = candidate(low)
        tokens = count(context)
        if tokens > limit:
            raise ValueError("Unable to fit prompt within context_window")
        return context, {"context_window": self.window, "input_tokens": tokens,
                         "reserved_output_tokens": self.output_tokens, "safety_margin": self.margin,
                         "token_count_method": self.encoding.name if self.encoding else "utf8_bytes",
                         "original_context_chars": original_chars, "retained_context_chars": len(context),
                         "truncation_strategy": self.strategy, "truncated": original_chars != len(context)}

    async def answer(self, query, context, **kwargs):
        context, _ = self.prepare_answer_context(query, context, kwargs.get("prompt_key"))
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.llm.get("api_key"), base_url=self.llm.get("base_url"))
        for attempt in range(max(1, int(self.options.get("max_retries", 3)))):
            try:
                response = await self._client.chat.completions.create(
                    model=self.llm.get("model", "gpt-4o-mini"),
                    messages=self._messages(query, context, kwargs.get("prompt_key")),
                    max_tokens=self.output_tokens, temperature=self.llm.get("temperature", 0),
                )
                text = response.choices[0].message.content or ""
                matches = list(re.finditer(r"(?:^|\n)\s*(?:#+\s*)?FINAL ANSWER\s*:\s*", text, re.I))
                return text[matches[-1].end():].strip() if matches else text.strip()
            except Exception:
                if attempt + 1 >= max(1, int(self.options.get("max_retries", 3))):
                    raise
                await asyncio.sleep(2 ** attempt)

    async def close(self):
        if self._client:
            await self._client.close()
