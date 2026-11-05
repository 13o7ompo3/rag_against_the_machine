import pytest
from src.evaluator import calculate_iou


def test_iou_identical_ranges():
    iou = calculate_iou(gt_start=10, gt_end=50, ret_start=10, ret_end=50)
    assert iou == pytest.approx(1.0)


def test_iou_disjoint_ranges():
    iou = calculate_iou(gt_start=0, gt_end=20, ret_start=30, ret_end=50)
    assert iou == 0.0


def test_iou_adjacent_boundary_touching():
    # End of first range touches start of second range -> intersection is 0
    iou = calculate_iou(gt_start=0, gt_end=25, ret_start=25, ret_end=50)
    assert iou == 0.0


def test_iou_partial_overlap():
    # GT: [0, 30] (len=30), RET: [15, 45] (len=30)
    # Intersection: [15, 30] (len=15)
    # Union: 30 + 30 - 15 = 45
    # IoU: 15 / 45 = 1/3
    iou = calculate_iou(gt_start=0, gt_end=30, ret_start=15, ret_end=45)
    assert iou == pytest.approx(1.0 / 3.0)


def test_iou_subset_range():
    # GT: [0, 40] (len=40), RET: [10, 30] (len=20)
    # Intersection: [10, 30] (len=20)
    # Union: 40 + 20 - 20 = 40
    # IoU: 20 / 40 = 0.5
    iou = calculate_iou(gt_start=0, gt_end=40, ret_start=10, ret_end=30)
    assert iou == pytest.approx(0.5)


def test_iou_zero_length_range():
    iou = calculate_iou(gt_start=10, gt_end=10, ret_start=10, ret_end=20)
    assert iou == 0.0

