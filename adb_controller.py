import atexit
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Optional
import cv2
import numpy as np

if sys.platform == "win32":
    import atexit
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)
    atexit.register(ctypes.windll.winmm.timeEndPeriod, 1)

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ADBController:
    """Обертка над ADB для взаимодействия с Android-устройством с поддержкой H.264 видеострима."""

    SCRCPY_VERSION = "2.4"
    SCRCPY_URL = "https://github.com/Genymobile/scrcpy/releases/download/v2.4/scrcpy-server-v2.4"
    SCRCPY_DEVICE_PATH = "/data/local/tmp/scrcpy-server.jar"

    def __init__(
        self,
        device_serial: Optional[str] = None,
        enable_stream: bool = True,
        enable_control: bool = True,
        max_size: int = 0,
        bitrate: Optional[int] = None,
        max_fps: Optional[int] = None,
    ):
        """
        :param device_serial: Серийный номер / IP:port ADB-устройства. Если None, берется из ADB_DEVICE_SERIAL или автодетектится.
        :param enable_stream: Включить ли аппаратный H.264 видеострим через scrcpy.
        :param enable_control: Включить ли прямой Scrcpy Binary Control Socket (sub-millisecond tap injection).
        :param max_size: Максимальное измерение кадра (0 = нативное разрешение экрана).
        :param bitrate: Битрейт потока в бит/с (по умолчанию из SCRCPY_BITRATE или 6 Mbps).
        :param max_fps: Максимальный FPS (по умолчанию из SCRCPY_FPS или 30).
        """
        self.adb_bin = self._find_adb_binary()
        devices = self._get_connected_devices()

        if device_serial is None:
            device_serial = os.getenv("ADB_DEVICE_SERIAL") or None

        if bitrate is None:
            env_bitrate = os.getenv("SCRCPY_BITRATE")
            bitrate = int(env_bitrate) if env_bitrate and env_bitrate.isdigit() else 6_000_000

        if max_fps is None:
            env_fps = os.getenv("SCRCPY_FPS")
            max_fps = int(env_fps) if env_fps and env_fps.isdigit() else 30

        if device_serial:
            if device_serial not in devices:
                print(f"[ADBController] Предупреждение: устройство '{device_serial}' не найдено среди активных устройств: {devices}")
            self.device_serial = device_serial
        else:
            if not devices:
                raise RuntimeError("Нет подключенных устройств ADB")
            self.device_serial = devices[0]

        print(f"[ADBController] Подключено к устройству: {self.device_serial}")

        # Параметры видеостриминга и управления
        self.enable_stream = enable_stream and HAS_PYAV
        self.enable_control = enable_control
        self.max_size = max_size
        self.bitrate = bitrate
        self.max_fps = max_fps

        self.screen_width = 1080
        self.screen_height = 2400

        self._stream_proc: Optional[subprocess.Popen] = None
        self._stream_socket: Optional[socket.socket] = None
        self._control_socket: Optional[socket.socket] = None
        self._touch_lock = threading.Lock()
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_running = False
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._local_port: Optional[int] = None

        if self.enable_stream or self.enable_control:
            self._start_stream()
            atexit.register(self.stop_stream)
        elif not HAS_PYAV and enable_stream:
            print("[ADBController] PyAV не установлен. Захват будет производиться через screencap.")

    @staticmethod
    def _find_adb_binary() -> str:
        """Поиск исполняемого файла adb в PATH или стандартной директории Android SDK."""
        if shutil.which("adb"):
            return "adb"

        # Проверка стандартного расположения на Windows
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            candidate = os.path.join(local_app_data, "Android", "Sdk", "platform-tools", "adb.exe")
            if os.path.isfile(candidate):
                return candidate

        return "adb"

    def _get_connected_devices(self) -> list[str]:
        """Парсинг вывода 'adb devices' с фильтрацией только активных устройств ('device')."""
        try:
            result = subprocess.run(
                [self.adb_bin, "devices"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка при вызове adb devices: {e}") from e

        active_devices = []
        for line in result.stdout.strip().splitlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                serial, status = parts[0], parts[1]
                if status == "device":
                    if serial not in active_devices:
                        active_devices.append(serial)

        return active_devices

    def _ensure_scrcpy_server(self) -> str:
        """Гарантирует наличие scrcpy-server.jar на хосте и заливает на устройство."""
        local_jar = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"scrcpy-server-v{self.SCRCPY_VERSION}.jar")
        if not os.path.exists(local_jar) or os.path.getsize(local_jar) == 0:
            print(f"[ADBController] Загрузка {os.path.basename(local_jar)} с GitHub...")
            urllib.request.urlretrieve(self.SCRCPY_URL, local_jar)
            print("[ADBController] scrcpy-server.jar успешно загружен.")

        # Проверяем наличие файла на устройстве
        check_cmd = [self.adb_bin, "-s", self.device_serial, "shell", "ls", "-l", self.SCRCPY_DEVICE_PATH]
        res = subprocess.run(check_cmd, capture_output=True, text=True)
        if str(os.path.getsize(local_jar)) not in res.stdout:
            print(f"[ADBController] Заливка {os.path.basename(local_jar)} на устройство...")
            push_cmd = [self.adb_bin, "-s", self.device_serial, "push", local_jar, self.SCRCPY_DEVICE_PATH]
            subprocess.run(push_cmd, capture_output=True, text=True, check=True)
            print("[ADBController] Сервер успешно залит на устройство.")

        return local_jar

    def _find_free_port(self) -> int:
        """Находит свободный локальный порт TCP."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _start_stream(self) -> None:
        """Инициализирует и запускает H.264 видеострим со scrcpy-server на устройстве."""
        try:
            self._ensure_scrcpy_server()

            self._local_port = self._find_free_port()
            forward_cmd = [
                self.adb_bin, "-s", self.device_serial,
                "forward", f"tcp:{self._local_port}", "localabstract:scrcpy"
            ]
            subprocess.run(forward_cmd, capture_output=True, text=True, check=True)

            control_flag = "true" if self.enable_control else "false"
            server_cmd = (
                f"CLASSPATH={self.SCRCPY_DEVICE_PATH} app_process / "
                f"com.genymobile.scrcpy.Server {self.SCRCPY_VERSION} "
                "tunnel_forward=true video=true audio=false "
                f"control={control_flag} cleanup=false raw_stream=true "
                f"video_bit_rate={self.bitrate} max_fps={self.max_fps}"
            )
            if self.max_size > 0:
                server_cmd += f" max_size={self.max_size}"

            self._stream_proc = subprocess.Popen(
                [self.adb_bin, "-s", self.device_serial, "shell", server_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Даем серверу инициализироваться (app_process запускает JVM)
            time.sleep(1.0)

            # Подключение к сокетам
            # При tunnel_forward=true сервер слушает один порт tcp:local_port:
            # Соединение 1: videoSocket
            # Соединение 2: controlSocket (если control=true)
            connected = False
            v_sock = None
            c_sock = None
            initial_chunk = b""

            for _ in range(40):
                if self._stream_proc.poll() is not None:
                    raise RuntimeError("Процесс scrcpy-server неожиданно завершился")

                try:
                    s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s1.settimeout(2.0)
                    s1.connect(("127.0.0.1", self._local_port))

                    if self.enable_control:
                        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s2.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        s2.settimeout(2.0)
                        s2.connect(("127.0.0.1", self._local_port))
                        c_sock = s2

                    chunk = s1.recv(4096)
                    if len(chunk) > 0:
                        initial_chunk = chunk
                        v_sock = s1
                        connected = True
                        break

                    s1.close()
                    if c_sock:
                        c_sock.close()
                        c_sock = None
                except (socket.error, ConnectionRefusedError):
                    time.sleep(0.1)

            if not connected or v_sock is None:
                raise RuntimeError("Не удалось установить сокеты со scrcpy-server")

            self._stream_socket = v_sock
            self._control_socket = c_sock
            if self._control_socket:
                print("[ADBController] Scrcpy Binary Control Socket успешно подключен (TCP_NODELAY=1)!")

            self._stream_running = True
            if self.enable_stream and HAS_PYAV:
                self._stream_thread = threading.Thread(
                    target=self._stream_decoder_loop,
                    args=(initial_chunk,),
                    daemon=True
                )
                self._stream_thread.start()

                # Ждем первый валидный кадр до 5 секунд
                t0 = time.time()
                while time.time() - t0 < 5.0:
                    with self._frame_lock:
                        if self._latest_frame is not None:
                            h, w = self._latest_frame.shape[:2]
                            self.screen_width = w
                            self.screen_height = h
                            print(f"[ADBController] Scrcpy H.264 видеострим успешно запущен! Разрешение: {w}x{h}, {self.max_fps} FPS.")
                            return
                    time.sleep(0.02)

                print("[ADBController] Предупреждение: Первый кадр не получен в течение 5 сек. Будет использован fallback screencap.")
            else:
                # PyAV отсутствует или видеострим отключен, но нужен сокет управления.
                # Запускаем фоновый дренаж видеосокета, чтобы буфер на устройстве не переполнялся.
                self._stream_thread = threading.Thread(
                    target=self._drain_stream_loop,
                    daemon=True
                )
                self._stream_thread.start()
                return
        except Exception as e:
            print(f"[ADBController] Ошибка запуска scrcpy видеострима/контроля: {e}. Переключение на fallback screencap.")
            self.stop_stream()

    def _drain_stream_loop(self) -> None:
        """Фоновый поток для пустого вычитывания видеопотока при отключенном декодере PyAV."""
        sock = self._stream_socket
        if not sock:
            return
        try:
            while self._stream_running:
                chunk = sock.recv(16384)
                if not chunk:
                    break
        except Exception:
            pass
        finally:
            self._stream_running = False

    def _stream_decoder_loop(self, initial_chunk: bytes = b"") -> None:
        """Фоновый поток для непрерывного декодирования H.264 пакетов и обновления latest_frame."""
        if not HAS_PYAV:
            return
        try:
            codec = av.CodecContext.create("h264", "r")
            sock = self._stream_socket
            if not sock:
                return

            if initial_chunk:
                for packet in codec.parse(initial_chunk):
                    for frame in codec.decode(packet):
                        img = frame.to_ndarray(format="bgr24")
                        with self._frame_lock:
                            self._latest_frame = img

            while self._stream_running:
                chunk = sock.recv(8192)
                if not chunk:
                    break

                for packet in codec.parse(chunk):
                    for frame in codec.decode(packet):
                        img = frame.to_ndarray(format="bgr24")
                        with self._frame_lock:
                            self._latest_frame = img
        except Exception:
            pass
        finally:
            self._stream_running = False

    def stop_stream(self) -> None:
        """Корректная остановка фонового стрима, сокета и процесса на устройстве."""
        self._stream_running = False

        if self._control_socket:
            try:
                self._control_socket.close()
            except Exception:
                pass
            self._control_socket = None

        if self._stream_socket:
            try:
                self._stream_socket.close()
            except Exception:
                pass
            self._stream_socket = None

        if self._stream_proc:
            try:
                self._stream_proc.terminate()
                self._stream_proc.wait(timeout=1.0)
            except Exception:
                try:
                    self._stream_proc.kill()
                except Exception:
                    pass
            self._stream_proc = None

        if self._local_port:
            try:
                subprocess.run(
                    [self.adb_bin, "-s", self.device_serial, "forward", "--remove", f"tcp:{self._local_port}"],
                    capture_output=True,
                    timeout=2.0
                )
            except Exception:
                pass
            self._local_port = None

        with self._frame_lock:
            self._latest_frame = None

    def get_frame(self) -> np.ndarray:
        """
        Возвращает актуальный снимок экрана.
        Если активен видеопоток — мгновенно возвращает копию последнего кадра (< 0.5–4 мс).
        В случае недоступности видеопотока выполняет fallback на screencap (~750–800 мс).
        """
        if self._stream_running:
            with self._frame_lock:
                if self._latest_frame is not None:
                    return self._latest_frame.copy()

        # Fallback на screencap
        return self._screencap_fallback()

    def _screencap_fallback(self) -> np.ndarray:
        """Получает текущий снимок экрана через медленный screencap (fallback)."""
        try:
            cmd = [self.adb_bin, "-s", self.device_serial, "exec-out", "screencap", "-p"]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
            )
            raw_bytes = proc.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Не удалось сделать снимок экрана через ADB: {e.stderr.decode('utf-8', errors='ignore')}") from e

        if not raw_bytes:
            raise RuntimeError("screencap вернул пустой буфер")

        frame = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Не удалось декодировать скриншот через OpenCV (cv2.imdecode вернул None)")

        return frame

    def inject_touch(
        self,
        x: int,
        y: int,
        action: int = 0,
        pointer_id: int = 0xFFFFFFFFFFFFFFFE,
        pressure: float = 1.0,
    ) -> bool:
        """
        Отправляет бинарный пакет INJECT_TOUCH_EVENT (32 байта) прямо в Scrcpy Control Socket.
        :param action: 0 = ACTION_DOWN, 1 = ACTION_UP, 2 = ACTION_MOVE
        :return: True при успешной отправке, False при недоступности сокета (нужен fallback).
        """
        if not self._control_socket:
            return False

        u16_pressure = 0xFFFF if pressure >= 1.0 else int(pressure * 65536.0)
        pkt = struct.pack(
            ">BBQiiHHHII",
            2,  # SC_CONTROL_MSG_TYPE_INJECT_TOUCH_EVENT
            action,
            pointer_id,
            int(x),
            int(y),
            self.screen_width,
            self.screen_height,
            u16_pressure,
            0,  # action_button
            0,  # buttons
        )
        try:
            with self._touch_lock:
                self._control_socket.sendall(pkt)
            return True
        except Exception as e:
            print(f"[ADBController] Сбой отправки в Control Socket: {e}. Сокет деактивирован.")
            try:
                self._control_socket.close()
            except Exception:
                pass
            self._control_socket = None
            return False

    def tap(self, x: int, y: int, delay: float = 0.05) -> None:
        """
        Выполняет клик (tap) по координатам (x, y).
        Если доступен Control Socket — мгновенно инжектирует DOWN -> UP (< 5 мс).
        При отсутствии сокета — штатный fallback на adb shell input tap.
        """
        if self._control_socket:
            ok_down = self.inject_touch(x, y, action=0)
            time.sleep(0.006)
            ok_up = self.inject_touch(x, y, action=1)
            if ok_down and ok_up:
                if delay > 0:
                    time.sleep(delay)
                return

        cmd = [self.adb_bin, "-s", self.device_serial, "shell", "input", "tap", str(x), str(y)]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            err_output = (e.stderr or "") + (e.stdout or "")
            if "SecurityException" in err_output or "INJECT_EVENTS" in err_output:
                hint = (
                    "\n[ОШИБКА РАЗРЕШЕНИЙ ADB / XIAOMI]\n"
                    "Android отклонил эмуляцию тапа (INJECT_EVENTS).\n"
                    "Для устройств Xiaomi (HyperOS / MIUI):\n"
                    "1. Откройте 'Настройки' -> 'Для разработчиков'.\n"
                    "2. Включите пункт 'Отладка по USB (Настройки безопасности)'\n"
                    "   (USB debugging (Security settings) — разрешить управление и ввод).\n"
                )
                print(hint)
            raise RuntimeError(f"Ошибка при выполнении tap({x}, {y}): {err_output.strip() or e}") from e

        if delay > 0:
            time.sleep(delay)

    def dismiss_popups(self, delay: float = 0.5) -> None:
        """Посылает событие нажатия кнопки 'Назад' (KEYCODE_BACK) для закрытия попапов и баннеров."""
        cmd = [self.adb_bin, "-s", self.device_serial, "shell", "input", "keyevent", "4"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except Exception as e:
            print(f"[ADBController] Предупреждение: не удалось отправить keyevent 4: {e}")

        if delay > 0:
            time.sleep(delay)

    def batch_double_taps_socket(
        self,
        coords: list[tuple[int, int]],
        hold_s: float = 0.010,
        interval_s: float = 0.042,
        gap_s: float = 0.025,
    ) -> bool:
        """
        Высокоскоростная аппаратная расстановка королев через Scrcpy Binary Control Socket:
        Каждая королева: DOWN -> hold -> UP -> interval -> DOWN -> hold -> UP
        Для каждой ячейки `i` используется уникальный `pointer_id = i`, что предотвращает
        ложную трактовку серии кликов как жеста свайпа (drag) движком Unity.
        Тайминги откалиброваны под 120 Гц: hold=10ms, interval=42ms, gap=25ms.
        :return: True если все тапы отправлены через сокет, False при сбое.
        """
        if not self._control_socket or not coords:
            return False

        try:
            for i, (x, y) in enumerate(coords):
                cell_pointer_id = i

                # Тап 1 (точка / крестик)
                if not self.inject_touch(x, y, action=0, pointer_id=cell_pointer_id):
                    return False
                time.sleep(hold_s)
                if not self.inject_touch(x, y, action=1, pointer_id=cell_pointer_id):
                    return False

                time.sleep(interval_s)

                # Тап 2 (королева)
                if not self.inject_touch(x, y, action=0, pointer_id=cell_pointer_id):
                    return False
                time.sleep(hold_s)
                if not self.inject_touch(x, y, action=1, pointer_id=cell_pointer_id):
                    return False

                if i < len(coords) - 1:
                    time.sleep(gap_s)
            return True
        except Exception:
            return False

    def batch_double_taps(
        self,
        coords: list[tuple[int, int]]
    ) -> None:
        """
        Выполняет всю серию быстрых дабл-тапов поячеечно:
        1. Если доступен Scrcpy Binary Control Socket — использует batch_double_taps_socket.
           При сбое сокета посреди передачи выбрасывает RuntimeError (слепой перезапуск с 0 запрещен).
        2. Фоллбэк на пакетный ADB shell input выполняется ТОЛЬКО если сокет изначально недоступен.
        """
        if not coords:
            return

        if self._control_socket:
            if not self.batch_double_taps_socket(coords):
                raise RuntimeError("[ADBController] Binary control socket injection failed mid-stream")
            return

        commands = []
        for i, (x, y) in enumerate(coords):
            commands.append(f"input tap {x} {y}; sleep 0.015; input tap {x} {y}")
            if i < len(coords) - 1:
                commands.append("sleep 0.005")

        chained_cmd = "; ".join(commands)
        cmd = [self.adb_bin, "-s", self.device_serial, "shell", chained_cmd]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            err_output = (e.stderr or "") + (e.stdout or "")
            if "SecurityException" in err_output or "INJECT_EVENTS" in err_output:
                hint = (
                    "\n[ОШИБКА РАЗРЕШЕНИЙ ADB / XIAOMI]\n"
                    "Android отклонил эмуляцию тапа (INJECT_EVENTS).\n"
                    "Для устройств Xiaomi (HyperOS / MIUI):\n"
                    "1. Откройте 'Настройки' -> 'Для разработчиков'.\n"
                    "2. Включите пункт 'Отладка по USB (Настройки безопасности)'\n"
                    "   (USB debugging (Security settings) — разрешить управление и ввод).\n"
                )
                print(hint)
            raise RuntimeError(f"Ошибка при пакетном выполнении тапов: {err_output.strip() or e}") from e



if __name__ == "__main__":
    controller = ADBController()
    print("Получение кадра с устройства...")
    t_start = time.perf_counter()
    frame = controller.get_frame()
    dt = (time.perf_counter() - t_start) * 1000.0
    print(f"Кадр успешно получен за {dt:.2f} мс! Shape: {frame.shape}")
    output_path = "debug_screen.png"
    cv2.imwrite(output_path, frame)
    print(f"Снимок сохранен в '{output_path}'")
    controller.stop_stream()

