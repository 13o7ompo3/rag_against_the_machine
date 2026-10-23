import torch
from transformers import AutoTokenizer, pipeline
import logging
from .models import MinimalSource
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=20)
def _read_file_cached(file_path: str) -> str:
    """Read and cache raw files in memory to avoid repeated disk I/O during batch runs."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_chunk_text(source: MinimalSource) -> str:
    """Slice the exact chunk text from the source file using its character indices."""
    try:
        content = _read_file_cached(source.file_path)
        start = max(0, source.first_character_index)
        end = min(len(content), source.last_character_index)
        return content[start:end]
    except Exception as exc:
        logger.warning(f"Failed to read source {source.file_path}: {exc}")
        return ""


class AnswerGenerator:
    """Generates answers from retrieved sources using a local causal LM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 256,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = pipeline(task="text-generation", model="Qwen/Qwen3-0.6B")
        self.model.generation_config.max_new_tokens = max_new_tokens
        self.max_new_tokens = max_new_tokens

    def _build_messages(
        self,
        question: str,
        sources: list[MinimalSource],
    ) -> list[dict[str, str]]:
        context_blocks: list[str] = []
        idx = 1
        for src in sources:
            chunk_text = extract_chunk_text(src).strip()
            if chunk_text:
                context_blocks.append(
                    f"[Source {idx}: {src.file_path}]\n{chunk_text}"
                )
                idx += 1
        context_str = (
            "\n\n---\n\n".join(context_blocks)
            if context_blocks
            else "No relevant context found."
        )

        system_prompt = (
            "You are a precise technical assistant for a database. "
            "Answer the user's question directly and concisely using ONLY "
            "the provided context. If the context does not contain enough "
            "information, state that clearly without guessing."
        )

        user_prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


    def generate_answer(
        self,
        question: str,
        sources: list[MinimalSource],
    ) -> str:
        if not sources:
            return "No relevant sources were retrieved to answer this question."

        messages = self._build_messages(question, sources)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        return self.model(prompt_text)["generated_text"]



