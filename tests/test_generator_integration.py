import os
import tempfile
from types import SimpleNamespace

from src.generator import AnswerGenerator
from src.models import MinimalSource, MinimalSearchResults


def test_generator_init_does_not_override_generation_config(monkeypatch):
    class DummyTokenizer:
        pass

    class DummyPipeline:
        def __init__(self):
            self.generation_config = SimpleNamespace(max_new_tokens=7)
            self.model = SimpleNamespace(
                config=SimpleNamespace(max_position_embeddings=2048)
            )

    monkeypatch.setattr(
        "src.generator.AutoTokenizer.from_pretrained",
        lambda model_name: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "src.generator.pipeline",
        lambda task, model, tokenizer: DummyPipeline(),
    )

    gen = AnswerGenerator(model_name="fake-model", max_new_tokens=12)

    assert gen.max_new_tokens == 12
    assert gen.context_window == 2048
    assert gen.model.generation_config.max_new_tokens == 12


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

        class DummyTokenizer:
            def encode(self, text):
                return text.split()

        gen = AnswerGenerator.__new__(AnswerGenerator)
        gen.tokenizer = DummyTokenizer()
        gen.context_window = 2048
        gen.max_new_tokens = 256

        messages = gen._build_messages(
            question="What host does redis connect to?",
            sources=[source],
        )

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert "technical assistant" in messages[0]["content"]

        user_content = messages[1]["content"]
        assert "Question: What host does redis connect to?" in user_content
        assert f"[Source 1: {temp_file}]" in user_content
        assert "redis-cluster" in user_content

        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Based on the provided context,"
    finally:
        os.remove(temp_file)


def test_generator_build_messages_empty_sources():
    class DummyTokenizer:
        def encode(self, text):
            return text.split()

    gen = AnswerGenerator.__new__(AnswerGenerator)
    gen.tokenizer = DummyTokenizer()
    gen.context_window = 2048
    gen.max_new_tokens = 256

    messages = gen._build_messages(
        question="Where is the config?",
        sources=[],
    )
    assert messages == []


def test_generator_generate_answer_mocked():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write("test content for generator")
        temp_file = f.name

    try:
        gen = AnswerGenerator.__new__(AnswerGenerator)
        gen.max_new_tokens = 64
        gen.context_window = 2048

        class DummyTokenizer:
            def encode(self, text):
                return text.split()

            def apply_chat_template(self, messages, tokenize=False, **kwargs):
                return "formatted prompt"

        class DummyModel:
            generation_config = SimpleNamespace(
                max_new_tokens=64,
                max_length=None,
            )

            def __call__(self, prompt, return_full_text=False, **kwargs):
                return [{"generated_text": "  Dummy generated answer  "}]

        gen.tokenizer = DummyTokenizer()
        gen.model = DummyModel()

        source = MinimalSource(
            file_path=temp_file,
            first_character_index=0,
            last_character_index=12,
        )
        answer = gen.generate_answer(
            question="What is the answer?",
            sources=[source],
        )
        assert answer == "Dummy generated answer"
    finally:
        os.remove(temp_file)


def test_generator_process_batch_mocked():
    gen = AnswerGenerator.__new__(AnswerGenerator)
    answers_map = {"Q1": "Answer 1", "Q2": "Answer 2"}
    gen.generate_answer = lambda question, sources: answers_map[question]

    results = [
        MinimalSearchResults(
            question_id="q1", question="Q1", retrieved_sources=[]
        ),
        MinimalSearchResults(
            question_id="q2", question="Q2", retrieved_sources=[]
        ),
    ]

    answers = gen.process_batch(results)
    assert len(answers) == 2
    assert answers[0].question_id == "q1"
    assert answers[0].answer == "Answer 1"
    assert answers[1].question_id == "q2"
    assert answers[1].answer == "Answer 2"


def test_generator_process_batch_handles_exception():
    gen = AnswerGenerator.__new__(AnswerGenerator)

    def failing_generate_answer(question, sources):
        raise RuntimeError("Inference error")

    gen.generate_answer = failing_generate_answer

    results = [
        MinimalSearchResults(
            question_id="q1", question="Q1", retrieved_sources=[]
        )
    ]

    answers = gen.process_batch(results)
    assert len(answers) == 1
    assert answers[0].answer == "Something went wrong !!"
