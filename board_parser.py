import cv2
import numpy as np
from typing import Tuple, List, Optional


class BoardParser:
    """Модуль компьютерного зрения для обнаружения и разбора игрового поля Queens Master."""

    def __init__(self, crop_ratio: float = 0.45):
        """
        :param crop_ratio: Доля центральной области клетки (0.0..1.0),
                           вырезаемая для оценки цвета без влияния границ.
        """
        self.crop_ratio = crop_ratio

    def find_board_bbox(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Локализует игровое поле на снимке экрана.
        Ищет крупный темный/цветной квадратный контур на светлом фоне.

        :param image: BGR-изображение экрана.
        :return: Кортеж (x, y, w, h).
        """
        h_img, w_img = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        total_area = float(w_img * h_img)

        # Пробуем несколько порогов яркости фона
        for thresh_val in [235, 240, 220]:
            mask = (gray < thresh_val).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidates = []
            for c in contours:
                x, y, bw, bh = cv2.boundingRect(c)
                area = bw * bh
                aspect = bw / float(bh) if bh > 0 else 0.0

                # Доска занимает > 15% экрана и почти квадратная
                if area > 0.15 * total_area and 0.85 <= aspect <= 1.15:
                    candidates.append((area, x, y, bw, bh))

            if candidates:
                candidates.sort(reverse=True, key=lambda item: item[0])
                _, x, y, bw, bh = candidates[0]
                return x, y, bw, bh

        raise RuntimeError("Не удалось локализовать игровое поле на изображении.")

    def detect_grid_size(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> int:
        """
        Динамически определяет размерность сетки N (N x N) в диапазоне от 5 до 15
        путем анализа проекций градиентов внутренних разделительных линий.

        :param image: BGR-изображение.
        :param bbox: (x, y, w, h) игрового поля.
        :return: Размерность N.
        """
        x, y, w, h = bbox
        crop = image[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Вычисляем производные Собеля для вертикальных и горизонтальных линий
        edges_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        edges_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

        prof_x = np.mean(edges_x, axis=0)
        prof_y = np.mean(edges_y, axis=1)

        best_n = 9
        best_score = -1.0

        for n in range(5, 16):
            step_x = w / float(n)
            step_y = h / float(n)
            win_x = max(3, int(step_x * 0.05))
            win_y = max(3, int(step_y * 0.05))

            # Оценка совпадения линий сетки с пиками градиента
            scores_x = []
            for k in range(1, n):
                lx = int(round(k * step_x))
                window = prof_x[max(0, lx - win_x):min(w, lx + win_x + 1)]
                if len(window) > 0:
                    scores_x.append(np.max(window))

            scores_y = []
            for k in range(1, n):
                ly = int(round(k * step_y))
                window = prof_y[max(0, ly - win_y):min(h, ly + win_y + 1)]
                if len(window) > 0:
                    scores_y.append(np.max(window))

            if scores_x and scores_y:
                score = (np.mean(scores_x) + np.mean(scores_y)) / 2.0
                if score > best_score:
                    best_score = score
                    best_n = n

        return best_n

    def extract_cells(
        self,
        bbox: Tuple[int, int, int, int],
        n: int
    ) -> List[List[Tuple[int, int]]]:
        """
        Вычисляет экранные центры ячеек игрового поля (чистая арифметика координат).

        :param bbox: (x, y, w, h) игрового поля.
        :param n: Размерность сетки N.
        :return: 2D-список центров клеток cell_centers_2d.
        """
        x, y, w, h = bbox
        cell_w = w / float(n)
        cell_h = h / float(n)

        cell_centers: List[List[Tuple[int, int]]] = []
        for r in range(n):
            row_centers = []
            for c in range(n):
                cx = int(round(x + (c + 0.5) * cell_w))
                cy = int(round(y + (r + 0.5) * cell_h))
                row_centers.append((cx, cy))
            cell_centers.append(row_centers)

        return cell_centers

    def parse_regions_by_borders(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        n: int
    ) -> np.ndarray:
        """
        Определяет регионы доски по физическим границам (Border Detection):
        - Разные регионы разделены жирными темными линиями (стенами).
        - Внутренние границы между клетками одного региона — тонкие и светлые.
        - Алгоритм:
          * Анализируются микро-патчи на перегородках между соседними ячейками.
          * Находится минимальная яркость перегородки.
          * Через DSU / Kruskal объединяются смежные ячейки в порядке убывания яркости
            (открытые проходы соединяются первыми) до тех пор, пока не останется ровно N компонент.
          * Инвариант: количество найденных регионов строго равно N.

        :param image: BGR-изображение кадра.
        :param bbox: (x, y, w, h) игрового поля.
        :param n: Размерность сетки N.
        :return: Матрица board размера (N, N) с каноническими ID регионов (0..N-1).
        """
        x, y, w, h = bbox
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        step_x = w / float(n)
        step_y = h / float(n)

        cx = [int(round(x + (c + 0.5) * step_x)) for c in range(n)]
        cy = [int(round(y + (r + 0.5) * step_y)) for r in range(n)]

        pw_h = max(2, int(round(step_x * 0.05)))
        ph_h = max(4, int(round(step_y * 0.20)))

        pw_v = max(4, int(round(step_x * 0.20)))
        ph_v = max(2, int(round(step_y * 0.05)))

        edges = []

        # Горизонтальные перегородки между (r, c) и (r, c+1)
        for r in range(n):
            yr = cy[r]
            for c in range(n - 1):
                xm = (cx[c] + cx[c + 1]) // 2
                patch = gray[
                    max(0, yr - ph_h):min(gray.shape[0], yr + ph_h + 1),
                    max(0, xm - pw_h):min(gray.shape[1], xm + pw_h + 1)
                ]
                val = float(np.min(patch))
                edges.append((val, r * n + c, r * n + (c + 1)))

        # Вертикальные перегородки между (r, c) и (r+1, c)
        for r in range(n - 1):
            for c in range(n):
                ym = (cy[r] + cy[r + 1]) // 2
                xr = cx[c]
                patch = gray[
                    max(0, ym - ph_v):min(gray.shape[0], ym + ph_v + 1),
                    max(0, xr - pw_v):min(gray.shape[1], xr + pw_v + 1)
                ]
                val = float(np.min(patch))
                edges.append((val, r * n + c, (r + 1) * n + c))

        # Сортировка по убыванию яркости: сначала объединяются самые светлые (свободные проходы)
        edges.sort(key=lambda item: item[0], reverse=True)

        parent = list(range(n * n))

        def find(i: int) -> int:
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        components = n * n

        for val, u, v in edges:
            if components == n:
                break
            ru = find(u)
            rv = find(v)
            if ru != rv:
                parent[ru] = rv
                components -= 1

        if components != n:
            raise ValueError(f"Не удалось выделить ровно N={n} регионов по границам (найдено {components})")

        # Каноническая перенумерация по первому появлению (0..N-1)
        canonical_board = np.zeros((n, n), dtype=int)
        mapping = {}
        next_id = 0

        for r in range(n):
            for c in range(n):
                root = find(r * n + c)
                if root not in mapping:
                    mapping[root] = next_id
                    next_id += 1
                canonical_board[r, c] = mapping[root]

        if len(np.unique(canonical_board)) != n:
            raise ValueError(f"Инвариант нарушен: ожидалось {n} регионов, получено {len(np.unique(canonical_board))}")

        return canonical_board

    def parse(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], Tuple[int, int, int, int]]:
        """
        Полный конвейер высокоскоростного парсинга доски:
        локализация -> определение N -> центры клеток -> Border Detection.

        :param image: BGR-изображение экрана.
        :return: (board_matrix, cell_centers_2d, bbox)
        """
        bbox = self.find_board_bbox(image)
        n = self.detect_grid_size(image, bbox)
        cell_centers = self.extract_cells(bbox, n)
        board = self.parse_regions_by_borders(image, bbox, n)
        return board, cell_centers, bbox

    def debug_visualize(
        self,
        image: np.ndarray,
        board: np.ndarray,
        cell_centers: List[List[Tuple[int, int]]],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        output_path: str = "debug_parsed_board.png"
    ) -> np.ndarray:
        """
        Отрисовывает визуализацию распознанной доски:
        рамку поля, центры ячеек и подписи ID регионов.

        :param image: Исходное изображение.
        :param board: Матрица регионов (N, N).
        :param cell_centers: 2D-список центров клеток (N, N).
        :param bbox: (x, y, w, h) игрового поля (опционально).
        :param output_path: Путь для сохранения изображения.
        :return: Аннотированное изображение.
        """
        annotated = image.copy()
        n = board.shape[0]

        # Отрисовка внешней границы доски
        if bbox is not None:
            bx, by, bw, bh = bbox
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 0, 255), 4)

        # Отрисовка центров и ID регионов
        for r in range(n):
            for c in range(n):
                cx, cy = cell_centers[r][c]
                region_id = int(board[r, c])

                # Точка в центре
                cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)

                # Текст с обводкой для четкой читаемости на любом фоне
                label = str(region_id)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                text_org = (cx - tw // 2, cy + th // 2)

                # Черная обводка
                cv2.putText(annotated, label, text_org, font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
                # Белый основной текст
                cv2.putText(annotated, label, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        cv2.imwrite(output_path, annotated)
        return annotated


if __name__ == "__main__":
    import os

    screen_path = "debug_screen.png"
    if not os.path.exists(screen_path):
        raise FileNotFoundError(f"Файл {screen_path} не найден для проверки.")

    print(f"Загрузка изображения '{screen_path}'...")
    img = cv2.imread(screen_path)
    if img is None:
        raise ValueError(f"Не удалось открыть изображение {screen_path}")

    parser = BoardParser()
    board, centers, bbox = parser.parse(img)
    n = board.shape[0]
    cell_size = bbox[2] / float(n)

    print(f"\nРезультаты парсинга игрового поля:")
    print(f"Bounding Box (x, y, w, h): {bbox}")
    print(f"Размерность сетки N: {n}x{n}")
    print(f"Размер клетки: {cell_size:.2f} px")
    print(f"Матрица регионов:")
    print(board)

    output_debug = "debug_parsed_board.png"
    parser.debug_visualize(img, board, centers, bbox, output_debug)
    print(f"\nВизуализация сохранена в '{output_debug}'")
