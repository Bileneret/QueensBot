import sys
import time
import numpy as np
from adb_controller import ADBController
from board_parser import BoardParser
from ui_locator import UILocator


def main():
    print("=" * 65)
    print("   BENCHMARK: Scrcpy H.264 Video Stream vs Board/UI Processing   ")
    print("=" * 65)

    ctrl = ADBController()
    parser = BoardParser()
    locator = UILocator()

    total_frames = 50
    latencies = []

    print(f"\n[1/3] Замер задержки доступа get_frame() ({total_frames} последовательных кадров)...")
    frames = []
    t_bench_start = time.perf_counter()

    for i in range(total_frames):
        t0 = time.perf_counter()
        frame = ctrl.get_frame()
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        latencies.append(dt_ms)
        frames.append(frame)
        time.sleep(0.01)  # 10ms интервал между запросами бота

    total_elapsed = time.perf_counter() - t_bench_start
    effective_fps = total_frames / total_elapsed

    avg_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)

    print(f"  -> Среднее время отдачи кадра: {avg_lat:.2f} мс")
    print(f"  -> Медиана:                    {median_lat:.2f} мс")
    print(f"  -> 95-й перцентиль:            {p95_lat:.2f} мс")
    print(f"  -> Мин / Макс:                 {min_lat:.2f} мс / {max_lat:.2f} мс")
    print(f"  -> Эффективный темп:           {effective_fps:.1f} FPS")

    last_frame = frames[-1]
    h, w = last_frame.shape[:2]
    print(f"\n[2/3] Проверка формата и резкости видеопотока...")
    print(f"  -> Размерность кадра: {w}x{h} (BGR)")

    print(f"\n[3/3] Тестирование алгоритмов распознавания на живом кадре видеопотока...")
    t_ui = time.perf_counter()
    state, target = locator.identify_screen(last_frame)
    dt_ui = (time.perf_counter() - t_ui) * 1000.0
    print(f"  -> UILocator.identify_screen: '{state}' (Target: {target}) за {dt_ui:.2f} мс")

    if state == "BOARD":
        t_parse = time.perf_counter()
        board, centers, bbox = parser.parse(last_frame)
        dt_parse = (time.perf_counter() - t_parse) * 1000.0
        print(f"  -> BoardParser.parse: размерность {board.shape[0]}x{board.shape[1]} за {dt_parse:.2f} мс")
    else:
        print(f"  -> Экран не является игровой доской (текущий экран: {state}).")
        try:
            bbox = parser.find_board_bbox(last_frame)
            print(f"  -> BBox доски: {bbox}")
        except Exception:
            print(f"  -> На текущем экране доска отсутствует (корректное поведение).")

    ctrl.stop_stream()

    print("\n" + "=" * 65)
    print("   РЕЗЮМЕ:")
    if avg_lat < 5.0:
        print(f"   [SUCCESS] Целевая задержка доступа < 5 мс ВЫПОЛНЕНА ({avg_lat:.2f} мс)!")
    else:
        print(f"   [WARN] Задержка доступа: {avg_lat:.2f} мс")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
