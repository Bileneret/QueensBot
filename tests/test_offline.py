"""Offline smoke tests: no Android device, no network, no hardware.

Covers the parts of the pipeline that are provably device-independent:

* solver (CP-SAT path and backtracking fallback) on a reference region matrix,
* BoardParser on a synthetic N x N frame (full parse pipeline),
* UILocator finite-state machine on synthetic frames.

Run:  pytest tests/test_offline.py -v
"""
import itertools

import cv2
import numpy as np
import pytest

from board_parser import BoardParser
from queens_solver import QueensSolver
from ui_locator import UILocator


REFERENCE_REGIONS = np.array([
    [0, 0, 0, 0, 0, 1, 1, 1],
    [2, 2, 0, 0, 1, 1, 1, 1],
    [2, 2, 2, 2, 1, 1, 1, 3],
    [2, 2, 2, 4, 4, 1, 3, 3],
    [5, 2, 4, 4, 4, 4, 3, 3],
    [5, 5, 5, 6, 4, 4, 7, 3],
    [5, 5, 6, 6, 6, 6, 7, 7],
    [5, 5, 6, 6, 6, 7, 7, 7],
], dtype=np.int32)


def _assert_valid_solution(board, queens):
    """Rows, columns and regions are exactly once each; Chebyshev distance >= 2."""
    n = board.shape[0]
    assert len(queens) == n, queens
    rows = [r for r, _ in queens]
    cols = [c for _, c in queens]
    regions = [int(board[r, c]) for r, c in queens]
    assert sorted(rows) == list(range(n))
    assert sorted(cols) == list(range(n))
    assert len(set(regions)) == n
    for (r1, c1), (r2, c2) in itertools.combinations(queens, 2):
        assert max(abs(r1 - r2), abs(c1 - c2)) >= 2


def _synthetic_frame(n=8, w=1080, h=2400, bx=40, by=600, bw=1000):
    """Render a Queens-like board frame: white bg, light cells, dark region walls."""
    bh = bw
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[by:by + bh, bx:bx + bw] = (245, 245, 245)
    cw = bw // n
    ch = bh // n
    for r in range(n):
        for c in range(n):
            x1, y1 = bx + c * cw, by + r * ch
            cv2.rectangle(img, (x1, y1), (x1 + cw, y1 + ch), (210, 210, 210), 1)
    for r in range(n):
        for c in range(n):
            x1, y1 = bx + c * cw, by + r * ch
            x2, y2 = x1 + cw, y1 + ch
            if c + 1 < n and REFERENCE_REGIONS[r, c + 1] != REFERENCE_REGIONS[r, c]:
                cv2.line(img, (x2, y1), (x2, y2), (40, 40, 40), 5)
            if r + 1 < n and REFERENCE_REGIONS[r + 1, c] != REFERENCE_REGIONS[r, c]:
                cv2.line(img, (x1, y2), (x2, y2), (40, 40, 40), 5)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (30, 30, 30), 6)
    return img


# ---------------------------------------------------------------- solver ----

def test_solver_reference_matrix():
    queens = QueensSolver(use_ortools=True).solve(REFERENCE_REGIONS)
    _assert_valid_solution(REFERENCE_REGIONS, queens)


def test_solver_backtracking_fallback():
    queens = QueensSolver(use_ortools=False).solve(REFERENCE_REGIONS)
    _assert_valid_solution(REFERENCE_REGIONS, queens)


def test_solver_rejects_non_square():
    with pytest.raises(ValueError):
        QueensSolver().solve(np.zeros((3, 4), dtype=np.int32))


# ----------------------------------------------------------- board parser ----

def test_parse_synthetic_board():
    img = _synthetic_frame(8)
    board, centers, bbox = BoardParser().parse(img)
    assert board.shape == (8, 8)
    assert len(np.unique(board)) == 8, "board must contain exactly N regions"
    assert len(centers) == 8 and all(len(row) == 8 for row in centers)
    _assert_valid_solution(board, QueensSolver().solve(board))


def test_parser_rejects_blank_frame():
    with pytest.raises(RuntimeError):
        BoardParser().find_board_bbox(np.full((600, 600, 3), 255, dtype=np.uint8))


# ------------------------------------------------------------ ui locator ----

def test_fsm_detects_board():
    state, target = UILocator().identify_screen(_synthetic_frame(8))
    assert state == "BOARD"
    assert target is not None


def test_fsm_unknown_on_ambiguous_screen():
    """A featureless gray screen must yield UNKNOWN with clicks forbidden."""
    frame = np.full((2400, 1080, 3), 128, dtype=np.uint8)
    state, target = UILocator().identify_screen(frame)
    assert state == "UNKNOWN"
    assert target is None


def test_modal_overlay_detection():
    locator = UILocator()
    assert not locator.is_modal_overlay_active(_synthetic_frame(8))
    assert locator.is_modal_overlay_active(np.full((2400, 1080, 3), 50, dtype=np.uint8))
