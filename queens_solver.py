import time
from typing import List, Tuple, Optional, Union
import cv2
import numpy as np

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


class QueensSolver:
    """Алгоритм поиска расстановки королев по правилам Star Battle / Queens."""

    def __init__(self, use_ortools: bool = True):
        self.use_ortools = use_ortools and ORTOOLS_AVAILABLE

    def solve(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """
        Находит координаты королев для заданной матрицы регионов board (N x N).

        :param board: 2D numpy массив (N, N), где каждое число — ID региона (0..N-1).
        :return: Список координат королев [(r0, c0), (r1, c1), ..., (r_{N-1}, c_{N-1})].
        :raises ValueError: Если решение не найдено.
        """
        if board.ndim != 2 or board.shape[0] != board.shape[1]:
            raise ValueError(f"Ожидалась квадратная матрица NxN, получена размерность {board.shape}")

        solution: Optional[List[Tuple[int, int]]] = None

        if self.use_ortools:
            solution = self._solve_cpsat(board)

        # Fallback на рекурсивный бэктрекинг
        if solution is None:
            solution = self._solve_backtracking(board)

        if solution is None:
            raise ValueError("Решение не найдено")

        return solution

    def _solve_cpsat(self, board: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Решение через CP-SAT (Google OR-Tools)."""
        n = board.shape[0]
        model = cp_model.CpModel()

        # Булевы переменные x[r, c] == 1, если в (r, c) стоит королева
        x = {}
        for r in range(n):
            for c in range(n):
                x[r, c] = model.NewBoolVar(f"x_{r}_{c}")

        # 1. Ровно одна королева в каждой строке
        for r in range(n):
            model.Add(sum(x[r, c] for c in range(n)) == 1)

        # 2. Ровно одна королева в каждом столбце
        for c in range(n):
            model.Add(sum(x[r, c] for r in range(n)) == 1)

        # 3. Ровно одна королева в каждом регионе
        unique_regions = np.unique(board)
        for reg_id in unique_regions:
            reg_cells = [(r, c) for r in range(n) for c in range(n) if board[r, c] == reg_id]
            model.Add(sum(x[r, c] for r, c in reg_cells) == 1)

        # 4. Расстояние Чебышёва >= 2 (королевы не касаются по 8 направлениям)
        for r in range(n):
            for c in range(n):
                # Проверяем соседей вперед по направлению, чтобы не дублировать ограничения
                for dr, dc in [(0, 1), (1, -1), (1, 0), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        model.Add(x[r, c] + x[nr, nc] <= 1)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            queens = [(r, c) for r in range(n) for c in range(n) if solver.Value(x[r, c]) == 1]
            queens.sort(key=lambda item: item[0])
            return queens

        return None

    def _solve_backtracking(self, board: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Рекурсивный поиск с возвратом (Backtracking)."""
        n = board.shape[0]
        used_cols = [False] * n
        used_regions = set()
        queens: List[Tuple[int, int]] = []

        def backtrack(r: int) -> bool:
            if r == n:
                return True

            for c in range(n):
                if used_cols[c]:
                    continue

                reg = int(board[r, c])
                if reg in used_regions:
                    continue

                # Проверка соседства: так как строки идут строго по порядку 0..n-1,
                # достаточно проверить, что королева в строке (r-1) не находится в соседней колонке
                if r > 0 and abs(c - queens[r - 1][1]) <= 1:
                    continue

                used_cols[c] = True
                used_regions.add(reg)
                queens.append((r, c))

                if backtrack(r + 1):
                    return True

                queens.pop()
                used_regions.remove(reg)
                used_cols[c] = False

            return False

        if backtrack(0):
            return queens

        return None

    def draw_solution(
        self,
        image: np.ndarray,
        cell_centers: Union[List[List[Tuple[int, int]]], List[Tuple[int, int]]],
        queens: List[Tuple[int, int]],
        output_path: str = "debug_solution.png"
    ) -> np.ndarray:
        """
        Отрисовывает найденные позиции королев на изображении.

        :param image: BGR-изображение экрана.
        :param cell_centers: 2D-список центров клеток [r][c] или плоский список.
        :param queens: Список координат королев [(r, c), ...].
        :param output_path: Путь сохранения файла с визуализацией.
        :return: Аннотированное изображение.
        """
        annotated = image.copy()

        for r, c in queens:
            if isinstance(cell_centers[0], list):
                cx, cy = cell_centers[r][c]
            else:
                # В случае одномерного списка центров
                n = int(np.sqrt(len(cell_centers)))
                cx, cy = cell_centers[r * n + c]

            # 1. Полупрозрачный фоновый кружок-бейдж
            overlay = annotated.copy()
            badge_radius = 42
            cv2.circle(overlay, (cx, cy), badge_radius, (20, 20, 20), -1)
            cv2.circle(overlay, (cx, cy), badge_radius, (0, 215, 255), 3, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            # 2. Отрисовка 5-конечной золотой звезды в центре
            star_radius = 32
            inner_radius = star_radius * 0.45
            star_pts = []
            for i in range(10):
                rad = star_radius if i % 2 == 0 else inner_radius
                angle = i * np.pi / 5.0 - np.pi / 2.0
                px = int(round(cx + rad * np.cos(angle)))
                py = int(round(cy + rad * np.sin(angle)))
                star_pts.append((px, py))

            star_poly = np.array(star_pts, np.int32)
            # Заливка золотистым цветом (BGR: 0, 215, 255)
            cv2.fillPoly(annotated, [star_poly], (0, 215, 255), lineType=cv2.LINE_AA)
            # Четкий контур звезды
            cv2.polylines(annotated, [star_poly], True, (0, 100, 180), 2, lineType=cv2.LINE_AA)

            # 3. Символ 'Q' в центре звезды
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize("Q", font, font_scale, thickness)
            tx = cx - tw // 2
            ty = cy + th // 2
            cv2.putText(annotated, "Q", (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        cv2.imwrite(output_path, annotated)
        return annotated


if __name__ == "__main__":
    import os
    from board_parser import BoardParser

    screen_file = "debug_screen.png"
    if not os.path.exists(screen_file):
        raise FileNotFoundError(f"Файл {screen_file} не найден.")

    print(f"Загрузка '{screen_file}' и парсинг доски...")
    img = cv2.imread(screen_file)
    parser = BoardParser()
    board, centers, bbox = parser.parse(img)

    solver = QueensSolver()
    print(f"Используемый движок: {'OR-Tools CP-SAT' if solver.use_ortools else 'Backtracking'}")

    start_time = time.perf_counter()
    queens_coords = solver.solve(board)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"\nРешение успешно найдено за {elapsed_ms:.2f} мс!")
    print(f"Координаты королев (строка, колонка): {queens_coords}")

    # Валидация решения
    n = board.shape[0]
    rows = [r for r, c in queens_coords]
    cols = [c for r, c in queens_coords]
    regions = [board[r, c] for r, c in queens_coords]
    print(f"Проверка строк: {len(set(rows))} / {n}")
    print(f"Проверка столбцов: {len(set(cols))} / {n}")
    print(f"Проверка регионов: {len(set(regions))} / {n}")

    output_solution = "debug_solution.png"
    solver.draw_solution(img, centers, queens_coords, output_solution)
    print(f"\nВизуализация решения сохранена в '{output_solution}'")
