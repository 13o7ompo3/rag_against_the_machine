import logging

from typing import Iterator
from transformers import AutoTokenizer, pipeline

from .models import MinimalSource, MinimalSearchResults, MinimalAnswer

logger = logging.getLogger(__name__)


def read_chunk(file_path: str, start: int, end: int) -> str:
    """Reads a character range from a text file from start to end index.

    Args:
        file_path: Path of the source file to read from.
        start: Starting character offset in the file.
        end: Ending character offset in the file.

    Returns:
        The extracted chunk string from the file.
    """
    if start < 0 or end < 0:
        raise ValueError("Indices must be non-negative integers.")
    if start > end:
        raise ValueError("Start index cannot be greater than end index.")
    if start == end:
        raise ValueError("Start and end indices cannot be the same.")

    length = end - start

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        f.read(start)
        return f.read(length)


def extract_chunk_text(source: MinimalSource) -> str:
    """Slice chunk text from a source file using character offsets.

    Args:
        source: MinimalSource containing file path and character range.

    Returns:
        The sliced chunk text, or empty string on read failure.
    """
    try:
        return read_chunk(
            source.file_path,
            source.first_character_index,
            source.last_character_index
            )
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
        """Initialize tokenizer and text-generation pipeline.

        Args:
            model_name: Hugging Face repository or local path for the model.
            max_new_tokens: Maximum number of tokens to generate per answer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = pipeline(task="text-generation", model=model_name)
        self.model.generation_config.max_new_tokens = max_new_tokens
        self.max_new_tokens = max_new_tokens

    def _build_messages(
        self,
        question: str,
        sources: list[MinimalSource],
    ) -> list[dict[str, str]]:
        """Construct system and user messages containing retrieved context.

        Args:
            question: The user's input query.
            sources: Retrieved MinimalSource objects to ground the response.

        Returns:
            A list of chat message dictionaries with role and content.
        """
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

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        """Applies chat template and formats messages into a model prompt.

        Args:
            messages: List of chat message dicts with role and content.

        Returns:
            Formatted chat prompt string.
        """
        return str(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def generate_answer(
        self,
        question: str,
        sources: list[MinimalSource],
    ) -> str:
        """Generate an answer for a single query using retrieved sources.

        Args:
            question: The user query string.
            sources: List of MinimalSource objects.

        Returns:
            Generated answer string.
        """
        if not sources:
            return (
                "No relevant sources were retrieved to answer "
                "this question."
            )

        messages = self._build_messages(question, sources)
        prompt_text = self._format_prompt(messages)

        outputs = self.model(
            prompt_text,
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )
        answer = outputs[0]["generated_text"]
        return answer.strip()

    def _generate_prompts(
        self, search_results: list[MinimalSearchResults]
    ) -> Iterator[str]:
        """Yields formatted prompt strings from search results.

        Args:
            search_results: List of search results to format.

        Yields:
            Formatted chat template strings.
        """
        for item in search_results:
            messages = self._build_messages(
                item.question,
                item.retrieved_sources,
            )
            yield self._format_prompt(messages)

    def process_batch(
        self, search_results: list[MinimalSearchResults], batch_size: int = 4
    ) -> list[MinimalAnswer]:
        """Generate answers for a batch of search results.

        Args:
            search_results: List of search result objects.
            batch_size: Inference batch size for the text generation pipeline.

        Returns:
            List of MinimalAnswer objects containing generated answers.
        """
        prompt_stream = self._generate_prompts(search_results)

        pipeline_outputs = self.model(
            prompt_stream,  # type: ignore[call-overload]
            batch_size=batch_size,
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )

        final_answers: list[MinimalAnswer] = []
        for item, output in zip(search_results, pipeline_outputs):
            if isinstance(output, list):
                generated_text = str(output[0]["generated_text"])
            else:
                generated_text = str(output["generated_text"])

            final_answers.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=generated_text.strip(),
                )
            )

        return final_answers
