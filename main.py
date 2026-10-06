import argparse
import sys
import time
from typing import Optional, Tuple, List
import numpy as np

if sys.platform == "win32":
    import atexit
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)
    atexit.register(ctypes.windll.winmm.timeEndPeriod, 1)

from adb_controller import ADBController
from board_parser import BoardParser
from queens_solver import QueensSolver
from ui_locator import UILocator


class QueensBot:
    """Полностью автономный высокоскоростной бот для головоломки Queens Master на Android."""

    def __init__(
        self,
        device_serial: Optional[str] = None,
        bitrate: Optional[int] = None,
        max_fps: Optional[int] = None,
    ):
        """
        :param device_serial: Серийный номер / IP:port ADB-устройства.
        :param bitrate: Битрейт потока в бит/с.
        :param max_fps: Максимальный FPS захвата.
        """
        self.adb = ADBController(device_serial=device_serial, bitrate=bitrate, max_fps=max_fps)
        self.parser = BoardParser()
        self.solver = QueensSolver()
        self.ui = UILocator()

    def _wait_for_button_to_disappear(self, timeout: float = 3.0) -> None:
        """Ожидает исчезновение кнопки следующего уровня после клика."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            time.sleep(0.08)
            try:
                frame = self.adb.get_frame()
                if self.ui.find_next_level_button(frame) is None:
                    return
            except Exception:
                return

    def transition_to_next_level(self, timeout: float = 6.0) -> bool:
        """
        Скоростной динамический переход на следующий уровень через детерминированную State Machine:
        1. Опрашивает экран каждые 0.15 с через identify_screen(frame).
        2. При состоянии VICTORY -> кликает по кнопке перехода слева от 'Домой' и ждет исчезновения.
        3. При состоянии POPUP_* -> кликает по детерминированной точке закрытия попапа.
        4. При состоянии UNKNOWN -> клики строго запрещены, ожидается стабилизация кадра.
        """
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            try:
                frame = self.adb.get_frame()
                state, target = self.ui.identify_screen(frame)

                # 1. Экран победы
                if state == "VICTORY" and target is not None:
                    bx, by = target
                    self.adb.tap(bx, by, delay=0.03)
                    self._wait_for_button_to_disappear()
                    return True

                # 2. Модальные окна блокираторов
                if state.startswith("POPUP_") and target is not None:
                    dx, dy = target
                    print(f"[QueensBot] Обнаружен {state}: закрытие по ({dx}, {dy})...")
                    self.adb.tap(dx, dy, delay=0.08)
                    time.sleep(0.3)

            except Exception:
                pass
            time.sleep(0.15)

        return False

    def wait_for_board(
        self,
        max_attempts: int = 40,
        interval: float = 0.02,
        require_stable_frames: int = 2
    ) -> Optional[Tuple[np.ndarray, List[List[Tuple[int, int]]], Tuple[int, int, int, int], np.ndarray]]:
        """
        Опрашивает экран через детерминированную State Machine до появления валидного игрового поля:
        1. BOARD -> мгновенный разбор поля через Border Detection и валидация через solver.
        2. VICTORY -> нажатие 'Следующий уровень' слева от 'Домой'.
        3. POPUP_* -> закрытие блокиратора в его детерминированной точке.
        4. UNKNOWN -> строгий запрет любых кликов, ожидание стабилизации.
        """
        last_board: Optional[np.ndarray] = None
        last_centers = None
        last_bbox = None
        last_frame = None
        stable_count = 0

        for attempt in range(1, max_attempts + 1):
            try:
                frame = self.adb.get_frame()
            except Exception:
                time.sleep(interval)
                continue

            state, target = self.ui.identify_screen(frame)

            # Состояние 1: Игровое поле (BOARD)
            if state == "BOARD":
                try:
                    board, centers, bbox = self.parser.parse(frame)
                    if require_stable_frames <= 1:
                        # Мгновенная валидация через solver: если поле решается без ошибок, оно 100% стабильно
                        self.solver.solve(board)
                        return board, centers, bbox, frame
                    else:
                        if last_board is not None and last_board.shape == board.shape and np.array_equal(last_board, board):
                            stable_count += 1
                        else:
                            stable_count = 1
                            last_board = board
                            last_centers = centers
                            last_bbox = bbox
                            last_frame = frame

                        if stable_count >= require_stable_frames:
                            return board, centers, bbox, frame
                except Exception:
                    stable_count = 0
                    last_board = None

            # Состояние 2: Экран победы (VICTORY)
            elif state == "VICTORY" and target is not None:
                bx, by = target
                self.adb.tap(bx, by, delay=0.03)
                self._wait_for_button_to_disappear()
                time.sleep(0.2)
                continue

            # Состояние 3: Блокирующие попапы (POPUP_*)
            elif state.startswith("POPUP_") and target is not None:
                dx, dy = target
                print(f"[QueensBot] Обнаружен {state}: закрытие по ({dx}, {dy})...")
                self.adb.tap(dx, dy, delay=0.08)
                time.sleep(0.3)
                continue

            # Состояние 4: UNKNOWN (клики строго запрещены)
            time.sleep(interval)

        return None

    def solve_current_level(self) -> bool:
        """
        Решает текущий открытый уровень с максимальной скоростью.
        """
        print("\n[QueensBot] Ожидание игрового поля...")
        board_data = self.wait_for_board(max_attempts=20, interval=0.15, require_stable_frames=2)
        if board_data is None:
            print("[QueensBot] Не удалось обнаружить игровое поле.")
            return False

        board, cell_centers, bbox, frame = board_data
        n = board.shape[0]

        # Решение
        try:
            t0 = time.perf_counter()
            queens = self.solver.solve(board)
            t_solve = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            print(f"[QueensBot] Ошибка алгоритма решения: {e}")
            return False

        print(f"[QueensBot] Поле {n}x{n} решено за {t_solve:.1f} мс.")

        # Высокоскоростная пакетная расстановка королев (< 800 мс)
        tap_coords = [cell_centers[r][c] for r, c in queens]
        t_tap_start = time.perf_counter()
        self.adb.batch_double_taps(tap_coords)
        t_tap = (time.perf_counter() - t_tap_start) * 1000.0

        print(f"[QueensBot] Все {n} королев выставлены за {t_tap:.0f} мс!")
        return True

    def run_loop(self, total_levels: int) -> None:
        """
        Автономный скоростной цикл прохождения серии уровней.
        """
        print(f"\n[QueensBot] Старт скоростной серии из {total_levels} уровней...")
        start_all = time.perf_counter()
        solved_count = 0

        try:
            for current in range(1, total_levels + 1):
                t_lvl_start = time.perf_counter()
                print(f"\n{'=' * 15} [ УРОВЕНЬ {current} ИЗ {total_levels} ] {'=' * 15}")

                # 1. Ожидание доски с мгновенной валидацией
                board_data = self.wait_for_board(max_attempts=60, interval=0.02, require_stable_frames=2)

                if board_data is None:
                    print("[QueensBot] Доска не найдена. Проверка попапов...")
                    try:
                        f = self.adb.get_frame()
                        has_board = False
                        try:
                            self.parser.find_board_bbox(f)
                            has_board = True
                        except Exception:
                            has_board = False

                        if not has_board:
                            close_pt = self.ui.find_popup_dismiss(f)
                            if close_pt:
                                self.adb.tap(close_pt[0], close_pt[1], delay=0.08)
                            elif self.ui.is_modal_overlay_active(f):
                                self.adb.dismiss_popups(delay=0.6)
                    except Exception:
                        pass

                    board_data = self.wait_for_board(max_attempts=30, interval=0.02, require_stable_frames=2)

                if board_data is None:
                    print(f"[QueensBot] Критический таймаут на уровне {current}. Остановка.")
                    break

                board, cell_centers, bbox, frame = board_data
                n = board.shape[0]

                # 2. Решение
                t0 = time.perf_counter()
                try:
                    queens = self.solver.solve(board)
                    t_solve = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    print(f"[QueensBot] Ошибка решения на уровне {current}: {e}")
                    break

                # 3. Скоростная пакетная расстановка королев
                tap_coords = [cell_centers[r][c] for r, c in queens]
                t_tap_start = time.perf_counter()
                self.adb.batch_double_taps(tap_coords)
                t_tap = (time.perf_counter() - t_tap_start) * 1000.0

                lvl_time = time.perf_counter() - t_lvl_start
                solved_count += 1
                print(f"[QueensBot] Уровень {current} пройден за {lvl_time:.2f} с! (Решение: {t_solve:.0f} мс, Тапы: {t_tap:.0f} мс)")

                # 4. Скоростной переход к следующему уровню
                if current < total_levels:
                    self.transition_to_next_level()
                else:
                    total_time = time.perf_counter() - start_all
                    avg_time = total_time / max(1, solved_count)
                    print(f"\n{'=' * 20} [ ИТОГОВАЯ СТАТИСТИКА ] {'=' * 20}")
                    print(f"Пройдено уровней: {solved_count} из {total_levels}")
                    print(f"Общее время: {total_time:.1f} с (в среднем {avg_time:.2f} с/уровень)")
                    print(f"{'=' * 56}\n")

        except KeyboardInterrupt:
            print("\n\n[QueensBot] Работа прервана пользователем (Ctrl+C).")
            total_time = time.perf_counter() - start_all
            print(f"[QueensBot] Завершено уровней: {solved_count}. Общее время: {total_time:.1f} с.\n")

    def run_watchdog(self) -> None:
        """
        Фоновый режим вотчера (ручной контроль интерфейса пользователем / --watch):
        - Опрос экрана каждые 0.15-0.2 сек.
        - ПОЛНЫЙ ИГНОР любых экранов победы и попапов (никаких кликов по UI).
        - Бот реагирует ТОЛЬКО на состояние BOARD.
        - Мгновенная расстановка королев через Border Detection и solver.
        - Ожидание исчезновения доски (поллинг раз в 0.2 сек) перед ожиданием следующего уровня.
        """
        print("\n" + "=" * 54)
        print("   [WATCHDOG] Ожидание появления игрового поля...   ")
        print("   (Режим наблюдения: реагирует ТОЛЬКО на BOARD)    ")
        print("   Для остановки нажмите Ctrl+C                    ")
        print("=" * 54 + "\n")

        solved_total = 0

        try:
            while True:
                time.sleep(0.020)
                try:
                    frame = self.adb.get_frame()
                except Exception:
                    continue

                # Проверяем состояние экрана: реагируем ТОЛЬКО на BOARD
                # Игнорируем модальные оверлеи и победные экраны
                if self.ui.is_modal_overlay_active(frame):
                    continue

                # 1. Первичная попытка распарсить игровое поле через Border Detection
                try:
                    board_1, centers_1, bbox_1 = self.parser.parse(frame)
                except Exception:
                    # Поле не найдено или регионы некорректны — пропускаем кадр без кликов
                    continue

                # Fast Debounce (45 мс): отсечка анимации вылета доски и стабилизация геометрии
                time.sleep(0.045)
                frame_2 = self.adb.get_frame()
                try:
                    board_2, centers_2, bbox_2 = self.parser.parse(frame_2)
                except Exception:
                    continue

                if not np.array_equal(board_1, board_2):
                    continue

                board = board_2
                centers = centers_2

                # 2. Мгновенная валидация и поиск решения
                t_solve_start = time.perf_counter()
                try:
                    queens = self.solver.solve(board)
                except Exception:
                    # Если доска не решается — пропускаем
                    continue

                t_solve = (time.perf_counter() - t_solve_start) * 1000.0

                # 3. Мгновенная пакетная расстановка королев (быстрый поячеечный дабл-тап)
                tap_coords = [centers[r][c] for r, c in queens]
                t_tap_start = time.perf_counter()
                self.adb.batch_double_taps(tap_coords)
                t_tap = (time.perf_counter() - t_tap_start) * 1000.0

                total_ms = (time.perf_counter() - t_solve_start) * 1000.0
                solved_total += 1
                n = board.shape[0]

                print(f"[QueensBot] Уровень {n}x{n} решен за {total_ms:.0f} мс! (Border Solve: {t_solve:.0f} мс, Ввод: {t_tap:.0f} мс)")
                print(f"[QueensBot] Всего решено: {solved_total}. Ожидание завершения уровня...")

                # 4. Ждем, пока доска не исчезнет с экрана (раз в 0.05 с)
                while True:
                    time.sleep(0.05)
                    try:
                        f_wait = self.adb.get_frame()
                        self.parser.parse(f_wait)
                    except Exception:
                        # Доска исчезла (переход / анимация победы)
                        break

                print("[QueensBot] Доска исчезла. Ожидание следующей доски...\n")

        except KeyboardInterrupt:
            print(f"\n[QueensBot] Фоновый режим остановлен. Всего решено уровней: {solved_total}.\n")


def main():
    parser = argparse.ArgumentParser(description="Queens Master Speedrun Bot")
    parser.add_argument("--single", action="store_true", help="Решить только текущий уровень")
    parser.add_argument("--auto", type=int, default=None, help="Авто-режим с указанием числа уровней")
    parser.add_argument("--watch", action="store_true", help="Фоновый режим вотчера (speedrun mode)")
    parser.add_argument("--serial", type=str, default=None, help="Серийный номер / IP:port ADB-устройства (или через ADB_DEVICE_SERIAL)")
    parser.add_argument("--bitrate", type=int, default=None, help="Битрейт scrcpy в бит/с (или через SCRCPY_BITRATE, по умолчанию 6000000)")
    parser.add_argument("--fps", type=int, default=None, help="FPS scrcpy потока (или через SCRCPY_FPS, по умолчанию 30)")
    args = parser.parse_args()

    bot = QueensBot(device_serial=args.serial, bitrate=args.bitrate, max_fps=args.fps)

    if args.single:
        bot.solve_current_level()
        return

    if args.auto is not None:
        if args.auto <= 0:
            print("Число уровней должно быть больше 0.")
            return
        bot.run_loop(args.auto)
        return

    if args.watch:
        bot.run_watchdog()
        return

    # Интерактивное меню
    print("\n" + "=" * 52)
    print("        Queens Master Speedrun Bot              ")
    print("=" * 52)
    print("1 — Решить только текущий уровень")
    print("2 — Серия уровней (авто-прохождение)")
    print("3 — Фоновый режим: 'Speedrun Watchdog'")
    print("=" * 52)

    try:
        choice = input("Ваш выбор (1, 2 или 3): ").strip()
        if choice == "1":
            bot.solve_current_level()
        elif choice == "2":
            try:
                levels = int(input("Введите число уровней: ").strip())
                if levels <= 0:
                    print("Число уровней должно быть > 0.")
                    return
                bot.run_loop(levels)
            except ValueError:
                print("Некорректный ввод числа.")
        elif choice == "3":
            bot.run_watchdog()
        else:
            print("Неизвестный пункт меню.")
    except KeyboardInterrupt:
        print("\n[QueensBot] Отмена.")


if __name__ == "__main__":
    main()
