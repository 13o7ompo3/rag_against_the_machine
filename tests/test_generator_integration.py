import os
import tempfile
import pytest
from src.generator import AnswerGenerator
from src.models import MinimalSource


def test_generator_build_messages_with_real_sources():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write(
            "import os\n\n"
            "def connect_redis():\n"
            "    return Redis(host='redis-cluster', port=6379)\n"
        )
        temp_file = f.name

    try:
        source = MinimalSource(
            file_path=temp_file,
            first_character_index=11,
            last_character_index=70,
        )

        gen = AnswerGenerator.__new__(AnswerGenerator)
        messages = gen._build_messages(
            question="What host does redis connect to?",
            sources=[source],
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "technical assistant" in messages[0]["content"]

        user_content = messages[1]["content"]
        assert "Question: What host does redis connect to?" in user_content
        assert f"[Source 1: {temp_file}]" in user_content
        assert "redis-cluster" in user_content
    finally:
        os.remove(temp_file)


def test_generator_build_messages_empty_sources():
    gen = AnswerGenerator.__new__(AnswerGenerator)
    messages = gen._build_messages(
        question="Where is the config?",
        sources=[],
    )
    user_content = messages[1]["content"]
    assert "No relevant context found." in user_content


def test_generator_generate_answer_mocked():
    gen = AnswerGenerator.__new__(AnswerGenerator)
    gen.max_new_tokens = 64

    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "formatted prompt"

    class DummyModel:
        def __call__(self, prompt, max_new_tokens, return_full_text):
            return [{"generated_text": "  Dummy generated answer  "}]

    gen.tokenizer = DummyTokenizer()
    gen.model = DummyModel()

    answer = gen.generate_answer(
        question="What is the answer?",
        sources=[MinimalSource(file_path="dummy.txt", first_character_index=0, last_character_index=5)],
    )
    assert answer == "Dummy generated answer"


def test_generator_process_batch_mocked():
    from src.models import MinimalSearchResults

    gen = AnswerGenerator.__new__(AnswerGenerator)
    gen.max_new_tokens = 64

    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "prompt"

    class DummyModel:
        def __call__(self, stream, batch_size, max_new_tokens, return_full_text):
            # Consume the stream and return outputs
            prompts = list(stream)
            return [[{"generated_text": f"Answer {i}"}] for i in range(len(prompts))]

    gen.tokenizer = DummyTokenizer()
    gen.model = DummyModel()

    results = [
        MinimalSearchResults(question_id="q1", question="Q1", retrieved_sources=[]),
        MinimalSearchResults(question_id="q2", question="Q2", retrieved_sources=[]),
    ]

    answers = gen.process_batch(results, batch_size=2)
    assert len(answers) == 2
    assert answers[0].question_id == "q1"
    assert answers[0].answer == "Answer 0"
    assert answers[1].question_id == "q2"
    assert answers[1].answer == "Answer 1"

