"""Exercise real Python authorization paths with substituted external interfaces."""

import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from src.bridge import ai_demo_bridge as bridge
from src.control import run_pick_place_task_v7 as runner
from src.planning import text_experiment_assistant as assistant
from src.planning import text_experiment_to_v8 as text_pipeline
from tests.offline_support import INSTRUCTION, TASK, TEMPLATE, OfflineTestCase, post


class ExecutionChainTests(OfflineTestCase):
    """HTTP -> real planner/parser -> real CLI gate -> mocked serial subprocess."""

    def setUp(self):
        super().setUp()
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.traj_root = self.root / "trajectories"
        self.enterContext(patch.dict(os.environ, {"MIMO_API_KEY": "offline-test-only"}))
        for module in (bridge, text_pipeline):
            self.enterContext(patch.object(module, "ROOT", self.root))
            self.enterContext(patch.object(module, "PY313", Path(sys.executable)))
            self.enterContext(patch.object(module, "YOLO_PYTHON", Path(sys.executable)))
        self.enterContext(patch.object(bridge, "LATEST_STATE", self.root / "latest.json"))
        self.enterContext(patch.object(bridge, "PUBLIC_VISION_DIR", self.root / "vision"))
        self.enterContext(patch.object(bridge, "ASSISTANT_PREVIEW", self.root / "vision/latest.jpg"))
        self.enterContext(patch.object(text_pipeline, "TRAJ_ROOT", self.traj_root))
        self.client = Mock()
        self.client.chat.return_value = self.response(TASK)
        self.enterContext(patch.object(text_pipeline, "MimoClient", return_value=self.client))
        self.final_executor = self.enterContext(patch.object(runner.subprocess, "run", return_value=Mock(returncode=0)))
        self.enterContext(patch.object(runner, "safe_print"))
        self.dispatch = self.enterContext(patch.object(bridge, "run_subprocess", side_effect=self.fake_external))
        self.enterContext(patch.object(text_pipeline, "run", side_effect=self.fake_external))
        self.serial_commands = []

    @staticmethod
    def response(task):
        return {"choices": [{"message": {"content": json.dumps(task)}}]}

    def fake_external(self, cmd, *, label, **kwargs):
        """No child processes: call Python entry points or provide fixture files."""
        script = Path(cmd[1]).name
        if script == "text_experiment_to_v8.py":
            with patch.object(sys, "argv", cmd[1:]), contextlib.redirect_stdout(io.StringIO()):
                try:
                    text_pipeline.main()
                except SystemExit as exc:
                    # Real subprocess exits become BridgeError in run_subprocess.
                    raise bridge.BridgeError(f"{label} failed: {exc}") from exc
        elif script == "vision_table_yolo_locator.py":
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"detections_in_table": [
                {"label": "beaker", "confidence": 0.99, "table_xy_norm": [0.1, 0.5], "base_xyz": [0.01, -0.4, 0.22]},
                {"label": "brown_bottle", "confidence": 0.99, "table_xy_norm": [0.5, 0.5], "base_xyz": [0.0, -0.4, 0.22]},
            ]}), encoding="utf-8")
        elif script == "build_pick_place_task_v7.py":
            name = cmd[cmd.index("--name") + 1]
            # Template and natural-language pipelines use different configured roots.
            task_dirs = (
                self.traj_root / f"task_{name}",
                self.root / "urdf_learn/playground/playground/public/trajectories" / f"task_{name}",
            )
            for task_dir in task_dirs:
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / "task_config.json").write_text(json.dumps({
                    "traj_a": "a.json", "traj_b": "b.json", "traj_c": "c.json",
                    "segment_frames": 2, "rate": 20.0, "gain": 0.5,
                }), encoding="utf-8")
        elif script == "run_pick_place_task_v7.py":
            with patch.object(sys, "argv", cmd[1:]):
                runner.main()
            serial_cmd = self.final_executor.call_args.args[0]
            self.assertEqual(Path(serial_cmd[1]).name, "serial_pick_and_place_v7.py")
            self.serial_commands.append(serial_cmd)
        else:
            self.fail(f"unexpected external operation: {script}")

    def test_normal_natural_language_planning_cannot_reach_hardware(self):
        status, payload = post("/api/assistant/plan", {"instruction": INSTRUCTION})
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["status"], "planned")
        self.assertIs(payload["task"]["confirm_before_run"], True)
        self.assertIs(payload["execution"]["build_only"], True)
        self.assertIs(payload["execution"]["confirm"], False)
        self.assertEqual(self.client.chat.call_count, 1)
        self.final_executor.assert_not_called()

    def test_plan_route_wins_over_execute_fields(self):
        status, payload = post("/api/assistant/plan", {
            "instruction": INSTRUCTION, "mode": "execute", "confirm": True,
        })
        self.assertEqual(status, 200, payload)
        self.assertIs(payload["execution"]["build_only"], True)
        self.assertIs(payload["execution"]["confirm"], False)
        self.final_executor.assert_not_called()

    def test_authorized_natural_language_execution_reaches_mocked_executor_once(self):
        status, payload = post("/api/assistant/execute", {"instruction": INSTRUCTION, "confirm": True})
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(len(self.serial_commands), 1)
        self.assertEqual(self.serial_commands[0].count("--confirm"), 1)

    def test_direct_plan_pipeline_never_promotes_confirm_to_execution(self):
        payload = bridge.run_mimo_pipeline(
            instruction=INSTRUCTION, mode="plan", camera_index=2, port_name="MOCK", confirm=True,
        )
        self.assertEqual(payload["status"], "planned")
        self.assertIs(payload["execution"]["build_only"], True)
        self.assertIs(payload["execution"]["confirm"], False)
        self.final_executor.assert_not_called()

    def test_template_plan_never_reaches_executor_even_with_confirm(self):
        status, payload = post("/api/assistant/plan", {"template_id": TEMPLATE, "confirm": True})
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["status"], "planned")
        self.final_executor.assert_not_called()
        self.client.chat.assert_not_called()

    def test_authorized_template_execution_reaches_mocked_executor_once(self):
        status, payload = post("/api/assistant/execute", {"template_id": TEMPLATE, "confirm": True})
        self.assertEqual(status, 200, payload)
        self.assertEqual(len(self.serial_commands), 1)
        self.assertIn("--confirm", self.serial_commands[0])
        self.client.chat.assert_not_called()

    def test_rejected_request_does_not_start_model_vision_or_execution(self):
        status, _ = post("/api/assistant/execute", {"instruction": INSTRUCTION, "task": TASK})
        self.assertEqual(status, 403)
        self.dispatch.assert_not_called()
        self.client.chat.assert_not_called()
        self.final_executor.assert_not_called()
        self.assertFalse((self.root / "latest.json").exists())

    def test_previous_plan_does_not_authorize_execution(self):
        status, payload = post("/api/assistant/plan", {"instruction": INSTRUCTION})
        self.assertEqual(status, 200, payload)
        status, _ = post("/api/assistant/execute", {"instruction": INSTRUCTION})
        self.assertEqual(status, 403)
        self.assertEqual(self.client.chat.call_count, 1)
        self.final_executor.assert_not_called()

    def test_confirmation_does_not_carry_over_to_next_request(self):
        status, payload = post("/api/assistant/execute", {"instruction": INSTRUCTION, "confirm": True})
        self.assertEqual(status, 200, payload)
        status, _ = post("/api/assistant/execute", {"instruction": INSTRUCTION})
        self.assertEqual(status, 403)
        self.assertEqual(self.client.chat.call_count, 1)
        self.assertEqual(self.final_executor.call_count, 1)

    def test_invalid_model_policy_blocks_even_authorized_execution(self):
        self.client.chat.return_value = self.response({**TASK, "confirm_before_run": "false"})
        status, payload = post("/api/assistant/execute", {"instruction": INSTRUCTION, "confirm": True})
        self.assertEqual(status, 500, payload)
        self.assertIn("confirm_before_run", payload["error"])
        self.final_executor.assert_not_called()
        self.assertFalse(self.traj_root.exists())

    def test_cli_without_confirm_only_builds(self):
        args = ["text_experiment_to_v8.py", INSTRUCTION, "--output", str(self.root / "cli.json")]
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(text_pipeline.main(), 0)
        self.final_executor.assert_not_called()

    def test_cli_build_only_dominates_confirm(self):
        args = ["text_experiment_to_v8.py", INSTRUCTION, "--build-only", "--confirm", "--output", str(self.root / "cli.json")]
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(text_pipeline.main(), 0)
        self.final_executor.assert_not_called()

    def test_planner_only_assistant_never_calls_executor(self):
        with patch.object(assistant, "MimoClient", return_value=self.client), patch.object(sys, "argv", [
            "text_experiment_assistant.py", INSTRUCTION, "--output", str(self.root / "assistant.json"),
        ]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(assistant.main(), 0)
        self.dispatch.assert_not_called()
        self.final_executor.assert_not_called()


class BottomLevelGuardTests(OfflineTestCase):
    """Run the existing serial guard using a fake serial client and tiny trajectories."""

    def setUp(self):
        super().setUp()
        serial_stub = ModuleType("serial")
        serial_stub.Serial = Mock(side_effect=AssertionError("real serial access forbidden"))
        self.enterContext(patch.dict(sys.modules, {"serial": serial_stub}))
        control_dir = str(Path(runner.__file__).parent)
        self.enterContext(patch.object(sys, "path", [control_dir, *sys.path]))
        self.control = importlib.import_module("serial_pick_and_place_v7")
        self.enterContext(patch.object(self.control, "load_trajectory", return_value=[[0.0] * 6, [0.01] * 6]))
        self.client = self.enterContext(patch.object(self.control, "V7SerialClient"))
        self.args = self.control.build_parser().parse_args([
            "--traj-a", "fake-a.json", "--traj-b", "fake-b.json", "--traj-c", "fake-c.json",
        ])

    def test_no_confirm_does_not_even_construct_serial_client(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.control.run(self.args), 0)
        self.client.assert_not_called()

    def test_confirm_reaches_only_substituted_serial_boundary(self):
        class FakeSerialBoundaryReached(Exception):
            pass

        self.args.confirm = True
        self.client.side_effect = FakeSerialBoundaryReached
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FakeSerialBoundaryReached):
            self.control.run(self.args)
        self.client.assert_called_once()
