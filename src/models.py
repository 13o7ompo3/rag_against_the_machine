import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """ A minimal representation of a source document.

    Attributes:
        file_path (str): The path to the source document.
        first_character_index (int): The index of the first character.
        last_character_index (int): The index of the last character.
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A representation of a question that has not yet been answered.

    Attributes:
        question_id (str): A unique identifier for the question.
        question (str): The text of the question.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A representation of a question that has been answered.

    Attributes:
        question_id (str): A unique identifier for the question.
        question (str): The text of the question.
        sources (List[MinimalSource]): A list of relevant sources.
        answer (str): The text of the answer.
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset for retrieval-augmented generation (RAG) tasks.

    Attributes:
        rag_questions (List[UnansweredQuestion | AnsweredQuestion]):
                      A list of questions.
    """
    rag_questions: List[UnansweredQuestion | AnsweredQuestion]


class MinimalSearchResults(BaseModel):
    """A minimal representation of search results.

    Attributes:
        question_id (str): A unique identifier for the question.
        question (str): The text of the question.
        retrieved_sources (List[MinimalSource]): A list of retrieved sources.
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """A minimal representation of an answer, extending MinimalSearchResults.

    Attributes:
        question_id (str): A unique identifier for the question.
        question (str): The text of the question.
        retrieved_sources (List[MinimalSource]): A list of retrieved sources.
        answer (str): The text of the answer.
    """
    answer: str


class StudentSearchResults(BaseModel):
    """A representation of search results for a student.

    Attributes:
        search_results (List[MinimalSearchResults]): A list of search results.
        k (int): The number of top results to return.
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """A representation of search results and an answer for a student.

    Attributes:
        search_results (List[MinimalAnswer]): A list of search results
                                              with answers.
        k (int): The number of top results to return.
    """
    search_results: List[MinimalAnswer]
    k: int
