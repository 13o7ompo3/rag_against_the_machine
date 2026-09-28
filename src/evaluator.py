import json
import logging
import sys
from src.models import StudentSearchResults, RagDataset, AnsweredQuestion

logger = logging.getLogger(__name__)


def calculate_iou(gt_start: int, gt_end: int,
                  ret_start: int, ret_end: int) -> float:
    """Calculates the Intersection over Union for two 1D character ranges.

    Args:
        gt_start: Start index of the ground truth range.
        gt_end: End index of the ground truth range.
        ret_start: Start index of the retrieved range.
        ret_end: End index of the retrieved range.

    Returns:
        The IoU value as a float between 0 and 1.
    """
    intersection_start = max(gt_start, ret_start)
    intersection_end = min(gt_end, ret_end)
    intersection_length = max(0, intersection_end - intersection_start)

    if intersection_length == 0:
        return 0.0

    gt_length = gt_end - gt_start
    ret_length = ret_end - ret_start
    union_length = gt_length + ret_length - intersection_length

    return intersection_length / union_length


def evaluate_results(student_results_path: str,
                     ground_truth_path: str) -> None:
    """
    Evaluate the student's search results against the ground truth dataset.

    Args:
        student_results_path: Path to the StudentSearchResults JSON file.
        ground_truth_path: Path to the RagDataset JSON file.
    """
    try:
        with open(student_results_path, 'r', encoding='utf-8') as f:
            student_results_data = json.load(f)
            if not isinstance(student_results_data, dict):
                raise ValueError("Student results JSON must be a dictionary.")
            student_results = StudentSearchResults(**student_results_data)
    except Exception as e:
        logger.error(f"Failed to load student results: {e}")
        sys.exit(1)

    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
            if not isinstance(ground_truth_data, dict):
                raise ValueError("Ground truth JSON must be a dictionary.")
            ground_truth = RagDataset(**ground_truth_data)
    except Exception as e:
        logger.error(f"Failed to load ground truth: {e}")
        sys.exit(1)

    ground_truth_map = {
        q.question_id: q for q in ground_truth.rag_questions
        if isinstance(q, AnsweredQuestion)
    }

    evaluated_questions = 0
    recall_results = {}

    for k in [1, 3, 5, 10]:
        recall_results[k] = 0.0
        evaluated_questions = 0
        total_recall = 0.0

        for result in student_results.search_results:
            gt_question = ground_truth_map.get(result.question_id)

            if not gt_question or not gt_question.sources:
                continue

            evaluated_questions += 1
            found_sources = 0

            for gt_source in gt_question.sources:
                is_found = False

                for ret_source in result.retrieved_sources[:k]:

                    if gt_source.file_path == ret_source.file_path:
                        iou = calculate_iou(
                            gt_start=gt_source.first_character_index,
                            gt_end=gt_source.last_character_index,
                            ret_start=ret_source.first_character_index,
                            ret_end=ret_source.last_character_index
                        )

                        if iou >= 0.05:
                            is_found = True
                            break

                if is_found:
                    found_sources += 1

            question_recall = found_sources / len(gt_question.sources)
            total_recall += question_recall

        if evaluated_questions > 0:
            recall_at_k = total_recall / evaluated_questions
        else:
            recall_at_k = 0.0
        recall_results[k] = recall_at_k

    print(f"Questions evaluated: {evaluated_questions}")
    for k, recall_at_k in recall_results.items():
        print(f"    Recall@{k}: {recall_at_k:.3f} ({recall_at_k * 100:.1f}%)")
