import os
import tempfile
import pytest
from src.models import MinimalSource


@pytest.fixture(autouse=True, scope="session")
def preserve_env_file():
    """Preserve the user's .env file across the test session."""
    env_path = ".env"
    original_content = None
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            original_content = f.read()

    yield

    if original_content is not None:
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(original_content)
    elif os.path.exists(env_path):
        os.remove(env_path)


@pytest.fixture
def sample_sources() -> list[MinimalSource]:
    """Provides a list of sample MinimalSource objects."""
    return [
        MinimalSource(
            file_path="doc1.py",
            first_character_index=0,
            last_character_index=50,
        ),
        MinimalSource(
            file_path="doc1.py",
            first_character_index=40,
            last_character_index=90,
        ),
        MinimalSource(
            file_path="doc2.md",
            first_character_index=10,
            last_character_index=60,
        ),
    ]


@pytest.fixture
def temp_raw_dir():
    """Creates a temporary directory with dummy raw files for chunker testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        py_file = os.path.join(tmp_dir, "test_code.py")
        with open(py_file, "w", encoding="utf-8") as f:
            f.write("def hello_world():\n    print('Hello world!')\n")

        md_file = os.path.join(tmp_dir, "readme.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Introduction\n\nThis is a sample markdown document.\n")

        ignored_file = os.path.join(tmp_dir, "data.csv")
        with open(ignored_file, "w", encoding="utf-8") as f:
            f.write("col1,col2\n1,2\n")

        yield tmp_dir
