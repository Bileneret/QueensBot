import os
import sys
import time
import cv2
from ui_locator import UILocator
from board_parser import BoardParser


def main():
    locator = UILocator()
    parser = BoardParser()

    dataset = [
        ("5343656559042569732.jpg", "POPUP_ROYAL_EXHIBITION", "Оффер тем / Royal Exhibition"),
        ("5343656559042569735.jpg", "POPUP_HARD_LEVEL", "Сложный уровень / Бустер"),
        ("5343656559042569734.jpg", "POPUP_CONFIRM_EXIT", "Подтверждение выхода / 'Вы уверены?'"),
        ("5343656559042569733.jpg", "POPUP_BUY_LIVES", "Покупка жизней / 'Продолжить?'"),
        ("5343656559042569282.jpg", "POPUP_PET_WINDOW", "Окно питомца"),
        ("5343656559042569736.jpg", "POPUP_DUEL_CHALLENGE", "Вызов на поединок"),
        ("5343656559042569288.jpg", "VICTORY", "Экран победы (синяя кнопка)"),
        ("5343656559042569289.jpg", "VICTORY", "Экран победы (оранжевая кнопка)"),
        ("5343656559042569280.jpg", "VICTORY", "Экран победы (фиолетовая кнопка)"),
        ("debug_screen.png", "BOARD", "Игровое поле (Border Detection)"),
    ]

    print("\n" + "=" * 80)
    print("      ТЕСТИРОВАНИЕ STATE MACHINE И РЕЕСТРА БЛОКИРАТОРОВ UI       ")
    print("=" * 80)

    passed_count = 0

    for fname, expected_state, desc in dataset:
        if not os.path.exists(fname):
            print(f"[SKIP] Файл не найден: {fname}")
            continue

        img = cv2.imread(fname)
        if img is None:
            print(f"[FAIL] Не удалось открыть: {fname}")
            continue

        t0 = time.perf_counter()
        pred_state, target = locator.identify_screen(img)
        dt = (time.perf_counter() - t0) * 1000.0

        is_ok = (pred_state == expected_state)
        if is_ok:
            passed_count += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"[{status:4s}] {fname:25s} -> {pred_state:22s} (Ожидалось: {expected_state:22s}) | {dt:5.2f} мс | Target: {target} | {desc}")

    print("=" * 80)
    total_tested = len([d for d in dataset if os.path.exists(d[0])])
    accuracy = (passed_count / max(1, total_tested)) * 100.0
    print(f"ИТОГ: Успешно {passed_count} / {total_tested} тестов ({accuracy:.1f}%)")
    print("=" * 80 + "\n")

    # Тестирование производительности Border Detection
    if os.path.exists("debug_screen.png"):
        print("Тестирование производительности Border Detection:")
        test_img = cv2.imread("debug_screen.png")
        bbox = parser.find_board_bbox(test_img)
        n = parser.detect_grid_size(test_img, bbox)

        bench_times = []
        for _ in range(100):
            t_start = time.perf_counter()
            b = parser.parse_regions_by_borders(test_img, bbox, n)
            bench_times.append((time.perf_counter() - t_start) * 1000.0)

        mean_dt = sum(bench_times) / len(bench_times)
        min_dt = min(bench_times)
        print(f"  -> Время работы parse_regions_by_borders: {mean_dt:.2f} мс (Min: {min_dt:.2f} мс) [Целевое: < 5 мс]")
        if mean_dt < 5.0:
            print("  -> [PASS] Производительность соответствует целевому нормативу (< 5 мс)!\n")
        else:
            print("  -> [FAIL] Превышено целевое время!\n")

    if passed_count != total_tested:
        sys.exit(1)


if __name__ == "__main__":
    main()
