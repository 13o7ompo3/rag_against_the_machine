import pytest
from src.hybrid import Source, merge_by_weighted_rank


def test_source_equality_and_hashing():
    src1 = Source(file_path="main.py", first_character_index=0, last_character_index=100)
    src2 = Source(file_path="main.py", first_character_index=0, last_character_index=100)
    src3 = Source(file_path="main.py", first_character_index=10, last_character_index=100)
    src4 = Source(file_path="other.py", first_character_index=0, last_character_index=100)

    assert src1 == src2
    assert hash(src1) == hash(src2)
    assert src1 != src3
    assert src1 != src4

    # Test set deduplication
    source_set = {src1, src2, src3}
    assert len(source_set) == 2


def test_merge_by_weighted_rank_favors_dual_matches():
    """Verify that an item found in BOTH lists is ranked higher than an item missing in one list."""
    doc_both = Source(file_path="doc_both.py", first_character_index=0, last_character_index=50)
    doc_only_1 = Source(file_path="doc_only_1.py", first_character_index=0, last_character_index=50)
    doc_only_2 = Source(file_path="doc_only_2.py", first_character_index=0, last_character_index=50)

    # list1 has: [doc_only_1 (rank 0), doc_both (rank 1)]
    # list2 has: [doc_only_2 (rank 0), doc_both (rank 1)]
    list1 = [doc_only_1, doc_both]
    list2 = [doc_only_2, doc_both]

    merged = merge_by_weighted_rank(list1, list2, weight1=1.0, weight2=1.0)

    # doc_both score: rank 1 + rank 1 = 2
    # doc_only_1 score: rank 0 + len(list2) (2) = 2 -> tied or both are top
    # Let's test with len = 3 to make the penalty crystal clear:
    extra = Source(file_path="extra.py", first_character_index=0, last_character_index=10)
    list1_large = [doc_only_1, doc_both, extra]
    list2_large = [doc_only_2, doc_both, extra]

    # doc_both score: 1 + 1 = 2
    # doc_only_1 score: 0 + 3 (len) = 3
    # doc_only_2 score: 3 (len) + 0 = 3
    merged_large = merge_by_weighted_rank(list1_large, list2_large, weight1=1.0, weight2=1.0)
    assert merged_large[0] == doc_both


def test_merge_by_weighted_rank_weights_preference():
    doc1 = Source(file_path="doc1.py", first_character_index=0, last_character_index=10)
    doc2 = Source(file_path="doc2.py", first_character_index=0, last_character_index=10)

    list1 = [doc1]
    list2 = [doc2]

    # Prioritize list1 by weighting weight1 higher than weight2
    # score(doc1) = 0 * weight1 + len(list2) * weight2 = 0.1
    # score(doc2) = len(list1) * weight1 + 0 * weight2 = 10.0
    merged = merge_by_weighted_rank(list1, list2, weight1=10.0, weight2=0.1)
    assert merged[0] == doc1

