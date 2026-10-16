"""Offline tests for device-independent ADBController logic.

Covers the transactional batch-tap path (a socket break mid-stream re-sends the
remaining cells through the adb shell fallback while already-delivered cells are
not repeated) and SHA-256 verification of the scrcpy-server jar. No device and
no network: the socket and subprocess layers are stubbed.

Run:  pytest tests/test_adb_offline.py -v
"""

import hashlib
import types

import pytest

from adb_controller import ADBController


class _FakeSocket:
    """Заглушка сокета: интересно только закрытие."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _controller() -> ADBController:
    """Экземпляр без __init__ (без adb, USB и фоновых потоков)."""
    ctrl = ADBController.__new__(ADBController)
    ctrl.adb_bin = "adb"
    ctrl.device_serial = "SERIAL"
    return ctrl


def _stub_inject(ctrl, failing_cell: int, failing_packet: int):
    """inject_touch, отказывающий ровно на одном пакете одной ячейки."""
    sent = []

    def inject(x, y, action=0, pointer_id=0, pressure=1.0):
        sent.append((x, y, pointer_id))
        packet_in_cell = sum(1 for s in sent if s[2] == pointer_id) - 1
        return not (pointer_id == failing_cell and packet_in_cell == failing_packet)

    ctrl.inject_touch = inject
    return sent


def test_batch_socket_returns_failure_cell_index(monkeypatch):
    ctrl = _controller()
    ctrl._control_socket = _FakeSocket()
    sent = _stub_inject(ctrl, failing_cell=2, failing_packet=1)
    monkeypatch.setattr("adb_controller.time.sleep", lambda *_: None)

    coords = [(10 * i, 20 * i) for i in range(5)]
    failed = ctrl.batch_double_taps_socket(coords)

    assert failed == 2, "должен вернуться индекс ячейки обрыва"
    assert max(p for _, _, p in sent) == 2, "ячейки после обрыва отправляться не должны"


def test_batch_falls_back_to_shell_for_remainder(monkeypatch):
    ctrl = _controller()
    broken = _FakeSocket()
    ctrl._control_socket = broken

    monkeypatch.setattr(ctrl, "batch_double_taps_socket", lambda coords: 2)

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return types.SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr("adb_controller.subprocess.run", fake_run)

    ctrl.batch_double_taps([(i, i) for i in range(5)])

    assert ctrl._control_socket is None and broken.closed, "битый сокет должен быть закрыт"
    chained = captured["cmd"][-1]
    assert "input tap 2 2" in chained, "ячейка обрыва досылается повторно"
    assert "input tap 1 1" not in chained, "уже переданные ячейки дублироваться не должны"
    assert "input tap 4 4" in chained, "хвост батча должен досылаться"


def test_batch_without_socket_uses_shell_for_all(monkeypatch):
    ctrl = _controller()
    ctrl._control_socket = None
    captured = {}
    monkeypatch.setattr("adb_controller.subprocess.run",
                        lambda cmd, **kw: captured.setdefault("cmd", cmd))

    ctrl.batch_double_taps([(1, 1), (2, 2)])

    chained = captured["cmd"][-1]
    assert "input tap 1 1" in chained and "input tap 2 2" in chained


def test_sha256_helper_matches_hashlib(tmp_path):
    payload = b"scrcpy-server-test-bytes"
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert ADBController._sha256_file(str(path)) == hashlib.sha256(payload).hexdigest()


def test_ensure_server_rejects_bad_download(monkeypatch, tmp_path):
    ctrl = _controller()
    ctrl.SCRCPY_SHA256 = "0" * 64
    jar = tmp_path / "scrcpy-server-v9.9.jar"
    monkeypatch.setattr(ctrl, "_local_jar_path", lambda: str(jar))

    def fake_urlretrieve(url, dest):
        with open(dest, "wb") as f:
            f.write(b"tampered payload")
        return (dest, None)

    monkeypatch.setattr("adb_controller.urllib.request.urlretrieve", fake_urlretrieve)

    with pytest.raises(RuntimeError, match="SHA-256"):
        ctrl._ensure_scrcpy_server()
    assert not jar.exists(), "непроверенный файл не должен оставаться"
    assert not (tmp_path / "scrcpy-server-v9.9.jar.part").exists()


def test_ensure_server_accepts_verified_local_file(monkeypatch, tmp_path):
    ctrl = _controller()
    payload = b"verified-jar-bytes"
    jar = tmp_path / "scrcpy-server-v9.9.jar"
    jar.write_bytes(payload)
    ctrl.SCRCPY_SHA256 = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(ctrl, "_local_jar_path", lambda: str(jar))

    def no_download(url, dest):  # pragma: no cover - вызываться не должен
        raise AssertionError("скачивание не должно происходить при валидном файле")

    monkeypatch.setattr("adb_controller.urllib.request.urlretrieve", no_download)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("adb_controller.subprocess.run", fake_run)

    result = ctrl._ensure_scrcpy_server()

    assert result == str(jar)
    assert any("ls" in part for part in calls[0]), "должна быть проверка файла на устройстве"
    assert any("push" in part for part in calls[-1]), "валидный файл должен заливаться"
