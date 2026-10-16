"""In-memory HTTP and tripwires against processes, networking and paid models."""

import io
import json
import unittest
from unittest.mock import patch

from src.bridge import ai_demo_bridge as bridge

TASK = {
    "intent": "pick_place",
    "object_label": "beaker",
    "source": "table_left",
    "place_target": "table_right",
    "confirm_before_run": True,
}
INSTRUCTION = "把左边的烧杯放到配置台右侧"
TEMPLATE = "demo_pick_brown_center_to_table_right"
WRONG_BOOLEANS = [None, 0, 1, 0.0, 1.0, "true", "false", "yes", "", [], {}, [True]]


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        self.tripwires = []
        for target in (
            "subprocess.Popen", "socket.socket", "urllib.request.urlopen", "os.system",
            "src.planning.mimo_client.MimoClient.chat",
        ):
            guard = patch(target, side_effect=AssertionError(f"offline test forbids {target}"))
            self.tripwires.append(guard.start())
            self.addCleanup(guard.stop)

    def tearDown(self):
        # Catch attempted I/O even if application error handling swallowed the exception.
        for guard in self.tripwires:
            guard.assert_not_called()


class MemoryConnection:
    def __init__(self, request):
        self.request = request
        self.response = bytearray()

    def makefile(self, *args, **kwargs):
        return io.BytesIO(self.request)

    def sendall(self, data):
        self.response.extend(data)


def post(path, body=None, *, raw=None, length=None):
    if raw is None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = (
        f"POST {path} HTTP/1.0\r\nHost: localhost\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(raw) if length is None else length}\r\n\r\n"
    ).encode("ascii") + raw
    connection = MemoryConnection(request)
    with patch.object(bridge.Handler, "log_message"):
        bridge.Handler(connection, ("127.0.0.1", 12345), object())
    head, payload = bytes(connection.response).split(b"\r\n\r\n", 1)
    return int(head.split()[1]), json.loads(payload.decode("utf-8"))
