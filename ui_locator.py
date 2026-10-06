import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from board_parser import BoardParser


class UILocator:
    """
    Детерминированная State Machine и локатор элементов пользовательского интерфейса
    на базе геометрических и структурных якорей интерфейса Queens Master.
    """

    def __init__(self):
        self.parser = BoardParser()

    # =========================================================================
    # Вспомогательные методы проверки оверлея
    # =========================================================================

    def is_modal_overlay_active(self, frame: np.ndarray) -> bool:
        """
        Проверяет, активен ли темный модальный оверлей попапа.
        В обычной игре фон экрана БЕЛЫЙ / СВЕТЛЫЙ (V > 180 в верхних углах [50:150, 50:150]).
        При открытии модального окна фон затемняется (V < 140).
        """
        h, w = frame.shape[:2]
        if h < 200 or w < 200:
            return False

        corner_tl = frame[50:150, 50:150]
        hsv_tl = cv2.cvtColor(corner_tl, cv2.COLOR_BGR2HSV)
        v_tl = float(np.mean(hsv_tl[:, :, 2]))

        if v_tl > 160.0:
            return False

        corner_tr = frame[50:150, max(0, w - 150):max(0, w - 50)]
        hsv_tr = cv2.cvtColor(corner_tr, cv2.COLOR_BGR2HSV)
        v_tr = float(np.mean(hsv_tr[:, :, 2]))

        return v_tl < 140.0 and v_tr < 140.0

    # =========================================================================
    # 1. Экран победы (Victory Screen)
    # =========================================================================

    def check_victory_screen(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Универсальная проверка экрана победы и локализация кнопки перехода.
        Цвет кнопки полностью игнорируется!
        Детект основан на структурных инвариантах:
        1. Нижний якорь: В самом низу экрана расположена БЕЛАЯ плашка действий,
           на которой справа ВСЕГДА находится статичная надпись 'Домой' (V > 210, S < 40).
        2. Верхний якорь: Белая карточка с золотой короной и надписью 'пройден'.
        3. Кнопка 'Следующий уровень': расположена слева от слова 'Домой' на этой же белой плашке.
           Вытянутая скругленная капсула (~0.35*W x ~0.06*H) с центром вокруг (~0.311*W, ~0.828*H).
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Нижний железобетонный якорь: белая плашка с надписью 'Домой' (V > 210, S < 40)
        home_roi = hsv[int(0.80 * h):int(0.86 * h), int(0.60 * w):int(0.95 * w)]
        home_is_white = (home_roi[:, :, 2].mean() > 210.0) and (home_roi[:, :, 1].mean() < 40.0)

        if not home_is_white:
            return None

        # Верхний якорь: золотая корона / 'пройден' (H in [15..35], S >= 100, V >= 150)
        gold_mask = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 150)
        top_gold = np.sum(gold_mask[int(0.05 * h):int(0.35 * h), int(0.10 * w):int(0.90 * w)])

        # Подтверждение экрана победы: верхняя корона или чистая белая подложка внизу
        if top_gold < 3000 and home_roi[:, :, 2].mean() < 230.0:
            return None

        # Локализация целевой кнопки перехода ("Следующий уровень")
        y_min = int(0.75 * h)
        y_max = int(0.90 * h)
        roi_hsv = hsv[y_min:y_max, 0:w]

        # Капсула на белом фоне
        btn_mask = (roi_hsv[:, :, 1] > 50) & (roi_hsv[:, :, 2] > 70)
        cnts, _ = cv2.findContours(btn_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            cx = bx + bw // 2
            cy = y_min + by + bh // 2
            if (
                0.20 * w <= bw <= 0.60 * w
                and 0.03 * h <= bh <= 0.10 * h
                and cx < 0.60 * w
            ):
                candidates.append((bw * bh, cx, cy))

        if candidates:
            candidates.sort(reverse=True, key=lambda item: item[0])
            return candidates[0][1], candidates[0][2]

        return int(round(0.311 * w)), int(round(0.828 * h))

    def find_next_level_button(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Обратная совместимость: возвращает координаты кнопки перехода на экране победы."""
        return self.check_victory_screen(frame)

    def is_victory_screen(self, frame: np.ndarray) -> bool:
        """Проверяет, является ли экран экраном победы."""
        return self.check_victory_screen(frame) is not None

    # =========================================================================
    # 2. Реестр 7 блокираторов (Модальные окна)
    # =========================================================================

    def check_popup_royal_exhibition(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Попап 1: Оффер тем / Royal Exhibition (5343656559042569732.jpg).
        - Якорь: Темно-бордовая карточка на темном оверлее + желтая кнопка цены '109,99 грн.'.
        - Действие: Одиночный белый крестик 'X' в самом верхнем правом углу экрана снаружи карточки.
        """
        h, w = frame.shape[:2]
        if not self.is_modal_overlay_active(frame):
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        burgundy = ((hsv[:, :, 0] <= 15) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 30) & (hsv[:, :, 2] <= 150)
        burgundy_cnt = np.sum(burgundy[int(0.20 * h):int(0.60 * h), int(0.10 * w):int(0.90 * w)])

        yellow_mask = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 150) & (hsv[:, :, 2] >= 150)
        yellow_btn = np.sum(yellow_mask[int(0.60 * h):int(0.85 * h), int(0.20 * w):int(0.80 * w)])

        if burgundy_cnt < 50000 or yellow_btn < 15000:
            return None

        # Белый крестик в самом верхнем правом углу экрана снаружи карточки
        roi = frame[0:int(0.15 * h), int(0.85 * w):w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        cross_mask = (gray_roi > 200).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(cross_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if 15 <= bw <= 70 and 15 <= bh <= 70:
                return int(0.85 * w) + bx + bw // 2, by + bh // 2

        return int(round(0.939 * w)), int(round(0.079 * h))

    def check_popup_duel_challenge(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Попап 6: Вызов на поединок (5343656559042569736.jpg).
        - Якорь: Карточка вызова со смайликом в красной рамке и золотой короной сверху +
                 круглая темная кнопка с белым крестиком (X) строго под белой карточкой.
        - Действие: Круглая темная кнопка с белым крестиком (X) строго под белой карточкой.
        """
        h, w = frame.shape[:2]
        if not self.is_modal_overlay_active(frame):
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        white_card = np.sum(gray[int(0.25 * h):int(0.65 * h), int(0.10 * w):int(0.90 * w)] > 220)

        # Белая карточка должна быть четко выражена
        if white_card < 200000:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 150) & (hsv[:, :, 2] >= 100)
        red_frame = np.sum(red_mask[int(0.20 * h):int(0.60 * h), int(0.20 * w):int(0.80 * w)])

        gold_mask = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 150) & (hsv[:, :, 2] >= 150)
        gold_crown = np.sum(gold_mask[int(0.15 * h):int(0.35 * h), int(0.30 * w):int(0.70 * w)])

        # Проверка круглого темного крестика строго под белой карточкой (y in 0.65*h..0.85*h)
        y_min = int(0.65 * h)
        y_max = int(0.85 * h)
        x_min = int(0.40 * w)
        x_max = int(0.60 * w)
        roi_gray = gray[y_min:y_max, x_min:x_max]

        # В центре круглого крестика белый символ (V > 230), а вокруг темный оверлей
        cx_target = int(round(0.499 * w))
        cy_target = int(round(0.751 * h))
        has_cross_in_center = (frame[cy_target, cx_target].mean() > 220.0)

        if (red_frame > 4000 and gold_crown > 2000) and (has_cross_in_center or white_card > 350000):
            # Проверяем, что это именно поединок, а не покупка жизней (нет красных сердец в ряд)
            # В покупке жизней нет золотой короны сверху на белой карточке (gold_crown < 500)
            if gold_crown > 2000:
                circles = cv2.HoughCircles(
                    roi_gray,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=50,
                    param1=50,
                    param2=25,
                    minRadius=int(0.02 * w),
                    maxRadius=int(0.06 * w)
                )
                if circles is not None:
                    return x_min + int(circles[0][0][0]), y_min + int(circles[0][0][1])
                return cx_target, cy_target

        return None

    def check_popup_confirm_exit(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Попап 3: Подтверждение выхода / 'Вы уверены?' (5343656559042569734.jpg).
        - Якорь: Желтый треугольный знак [ ! ] на белой карточке + грустный хомяк.
        - Действие: Темный крестик 'X' в верхнем правом углу белой карточки.
        """
        h, w = frame.shape[:2]
        if not self.is_modal_overlay_active(frame):
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        white_card = np.sum(gray[int(0.25 * h):int(0.65 * h), int(0.10 * w):int(0.90 * w)] > 220)
        if white_card < 250000:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Желтый треугольник (H in [15..35], S >= 150, V >= 150) внутри белой карточки
        yellow_mask = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 150) & (hsv[:, :, 2] >= 150)
        yellow_sign = np.sum(yellow_mask[int(0.30 * h):int(0.55 * h), int(0.30 * w):int(0.70 * w)])

        if yellow_sign < 5000:
            return None

        # Темный крестик в правом верхнем углу белой карточки
        roi = frame[int(0.25 * h):int(0.35 * h), int(0.75 * w):int(0.92 * w)]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if 15 <= bw <= 60 and 15 <= bh <= 60:
                return int(0.75 * w) + bx + bw // 2, int(0.25 * h) + by + bh // 2

        return int(round(0.827 * w)), int(round(0.298 * h))

    def check_popup_buy_lives(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Попап 4: Покупка жизней / 'Продолжить?' (5343656559042569733.jpg).
        - Якорь: 3 красных сердца в ряд внутри белой карточки + кнопка 'Играть дальше 50'.
        - Действие: Темный крестик 'X' в верхнем правом углу белой карточки.
        """
        h, w = frame.shape[:2]
        if not self.is_modal_overlay_active(frame):
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        white_card = np.sum(gray[int(0.25 * h):int(0.65 * h), int(0.10 * w):int(0.90 * w)] > 220)
        if white_card < 250000:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Красные сердца в ряд (H in [0..10] или [170..180], S >= 150, V >= 100)
        red_mask = ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 150) & (hsv[:, :, 2] >= 100)
        red_hearts = np.sum(red_mask[int(0.30 * h):int(0.55 * h), int(0.20 * w):int(0.80 * w)])

        # Проверяем отсутствие золотой короны сверху (отличие от дуэли)
        gold_mask = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 150) & (hsv[:, :, 2] >= 150)
        gold_crown = np.sum(gold_mask[int(0.15 * h):int(0.35 * h), int(0.30 * w):int(0.70 * w)])

        if red_hearts < 6000 or gold_crown > 1500:
            return None

        # Темный крестик в правом верхнем углу белой карточки
        roi = frame[int(0.25 * h):int(0.35 * h), int(0.75 * w):int(0.92 * w)]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if 15 <= bw <= 60 and 15 <= bh <= 60:
                return int(0.75 * w) + bx + bw // 2, int(0.25 * h) + by + bh // 2

        return int(round(0.827 * w)), int(round(0.298 * h))

    def check_popup_pet_window(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Попап 5: Окно питомца (5343656559042569282.jpg).
        - Якорь: Красная миска '0/10' с молнией вверху + грустный хомяк + кнопки покупки еды.
        - Действие: Белая надпись 'Нет, спасибо' в самом низу экрана на темной подложке.
        """
        h, w = frame.shape[:2]
        if not self.is_modal_overlay_active(frame):
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # В окне питомца нет огромной цельной белой карточки по центру
        white_card = np.sum(gray[int(0.25 * h):int(0.65 * h), int(0.10 * w):int(0.90 * w)] > 220)
        if white_card > 250000:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Красная миска вверху (H in [0..10] или [170..180], S >= 150, V >= 100)
        red_mask = ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 150) & (hsv[:, :, 2] >= 100)
        red_bowl = np.sum(red_mask[int(0.15 * h):int(0.35 * h), int(0.30 * w):int(0.70 * w)])

        # Проверка наличия кнопок покупки еды / миски
        if red_bowl < 2500:
            return None

        # Надпись 'Нет, спасибо' внизу экрана
        roi = frame[int(0.78 * h):int(0.88 * h), int(0.30 * w):int(0.70 * w)]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_roi, 160, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        dil = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw > 100:
                return int(0.30 * w) + bx + bw // 2, int(0.78 * h) + by + bh // 2

        return int(round(0.501 * w)), int(round(0.837 * h))

    def check_popup_hard_level(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Попап 2: Сложный уровень / Бустер (5343656559042569735.jpg).
        - Якорь: Текст сложности + синяя кнопка бустера 'Авто X Бесплатно'.
        - Действие: Белая надпись 'Нет, спасибо' строго под синей кнопкой.
        """
        h, w = frame.shape[:2]
        if not self.is_modal_overlay_active(frame):
            return None

        # В сложного уровня нет белой карточки
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        white_card = np.sum(gray[int(0.25 * h):int(0.65 * h), int(0.10 * w):int(0.90 * w)] > 220)
        if white_card > 100000:
            return None

        # Проверяем синюю кнопку бустера (B > 180, R < 110, Y in 0.65*h..0.75*h)
        roi_btn = frame[int(0.65 * h):int(0.75 * h), int(0.15 * w):int(0.85 * w)]
        blue_pix = (roi_btn[:, :, 0] > 180) & (roi_btn[:, :, 2] < 110)

        # Проверяем, что нет миски питомца
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 150) & (hsv[:, :, 2] >= 100)
        red_bowl = np.sum(red_mask[int(0.15 * h):int(0.35 * h), int(0.30 * w):int(0.70 * w)])

        if np.sum(blue_pix) < 30000 or red_bowl > 2000:
            return None

        # Локализация надписи 'Нет, спасибо' строго под синей кнопкой (y in 0.72*h..0.80*h)
        roi = frame[int(0.72 * h):int(0.80 * h), int(0.25 * w):int(0.75 * w)]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_roi, 160, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        dil = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw > 100:
                return int(0.25 * w) + bx + bw // 2, int(0.72 * h) + by + bh // 2

        return int(round(0.501 * w)), int(round(0.759 * h))

    def check_popup_defeat(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Попап 7: Поражение / Закончились жизни (1000004296.jpg).
        - Якорь: Белый круг с разбитым сердцем по центру + оранжевая плашка 'Закончились жизни X'.
        - Действие: Крестик 'X' на правом краю оранжевой плашки.
        """
        h, w = frame.shape[:2]
        if not self.is_modal_overlay_active(frame):
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Оранжевая плашка (H in [8..25], S >= 140, V >= 140)
        orange_mask = (hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] >= 140) & (hsv[:, :, 2] >= 140)
        orange_bar = orange_mask[int(0.45 * h):int(0.75 * h), int(0.10 * w):int(0.90 * w)]

        if np.sum(orange_bar) < 15000:
            return None

        # Крестик на правом краю оранжевой плашки
        cnts, _ = cv2.findContours(orange_bar.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw > 0.40 * w:
                cx = int(0.10 * w) + bx + int(bw * 0.92)
                cy = int(0.45 * h) + by + bh // 2
                return cx, cy

        return int(round(0.850 * w)), int(round(0.600 * h))

    # =========================================================================
    # 3. Детерминированная State Machine
    # =========================================================================

    def identify_screen(self, frame: np.ndarray) -> Tuple[str, Optional[Tuple[int, int]]]:
        """
        Детерминированная State Machine: проверяет состояния экрана в строгом приоритете.

        Приоритет:
        1. BOARD — если parser.find_board_bbox() видит границы доски без затемнения (нет оверлея).
        2. VICTORY — если найден якорь экрана победы ('Домой' на белой плашке снизу / 'пройден').
           Возвращает координаты кнопки 'Следующий уровень' слева от 'Домой'.
        3. POPUP_* — последовательная проверка каждого из 7 блокираторов по их визуальным якорям.
           Возвращает точную точку закрытия.
        4. UNKNOWN — если ни одно состояние не подтвердилось на 100%, возвращается (None, None).
           Любые клики по экрану СТРОГО ЗАПРЕЩЕНЫ, скрипт ждет стабилизации кадра.

        :param frame: BGR-изображение экрана.
        :return: (state_name, target_coords) где target_coords это (x, y) или None.
        """
        # Шаг 1: BOARD (проверяем границы доски БЕЗ затемнения)
        if not self.is_modal_overlay_active(frame):
            try:
                bbox = self.parser.find_board_bbox(frame)
                return "BOARD", (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
            except Exception:
                pass

        # Шаг 2: VICTORY (экран победы)
        vic_coords = self.check_victory_screen(frame)
        if vic_coords is not None:
            return "VICTORY", vic_coords

        # Шаг 3: POPUP_* (последовательная проверка 7 блокираторов)
        # 1. Оффер тем (Royal Exhibition)
        pt = self.check_popup_royal_exhibition(frame)
        if pt is not None:
            return "POPUP_ROYAL_EXHIBITION", pt

        # 2. Вызов на поединок
        pt = self.check_popup_duel_challenge(frame)
        if pt is not None:
            return "POPUP_DUEL_CHALLENGE", pt

        # 3. Подтверждение выхода
        pt = self.check_popup_confirm_exit(frame)
        if pt is not None:
            return "POPUP_CONFIRM_EXIT", pt

        # 4. Покупка жизней
        pt = self.check_popup_buy_lives(frame)
        if pt is not None:
            return "POPUP_BUY_LIVES", pt

        # 5. Окно питомца
        pt = self.check_popup_pet_window(frame)
        if pt is not None:
            return "POPUP_PET_WINDOW", pt

        # 6. Сложный уровень / Бустер
        pt = self.check_popup_hard_level(frame)
        if pt is not None:
            return "POPUP_HARD_LEVEL", pt

        # 7. Поражение / Закончились жизни
        pt = self.check_popup_defeat(frame)
        if pt is not None:
            return "POPUP_DEFEAT", pt

        # Шаг 4: UNKNOWN — ни одно состояние не подтвердилось, клики строго запрещены!
        return "UNKNOWN", None

    def find_popup_dismiss(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Обратная совместимость: возвращает точку закрытия попапа через State Machine."""
        state, target = self.identify_screen(frame)
        if state.startswith("POPUP_"):
            return target
        return None
