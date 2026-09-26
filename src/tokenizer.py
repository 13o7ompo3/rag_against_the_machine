import re


def tokenize(text: str) -> list[str]:
    """
    Tokenizes the input text into a list of tokens.

    Args:
        text (str): The input text to be tokenized.
    """
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    text = text.lower()

    text = re.sub(r'[^a-z0-9]', ' ', text)
    return text.split()
