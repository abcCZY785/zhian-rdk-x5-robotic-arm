"""Consent must be independent, explicit, strictly typed, and fail closed."""

import json
from unittest.mock import Mock, patch

from src.bridge import ai_demo_bridge as bridge
from src.planning.task_planner import extract_json_object, plan_text_instruction
from src.planning.task_schema import PlannedTask
from src.safety.execution_authorization import (
    ExecutionConfirmationRequired, require_execution_confirmation,
)
from tests.offline_support import (
    INSTRUCTION, TASK, TEMPLATE, WRONG_BOOLEANS, OfflineTestCase, post,
)


class TaskPolicyTests(OfflineTestCase):
    def test_explicit_boolean_true_is_preserved(self):
        self.assertIs(PlannedTask.from_dict(TASK).confirm_before_run, True)

    def test_missing_confirmation_policy_is_rejected(self):
        payload = dict(TASK)
        del payload["confirm_before_run"]
        with self.assertRaisesRegex(ValueError, "confirm_before_run"):
            PlannedTask.from_dict(payload)

    def test_false_and_coerced_values_are_rejected(self):
        for value in [False, *WRONG_BOOLEANS]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PlannedTask.from_dict({**TASK, "confirm_before_run": value})

    def test_direct_construction_cannot_disable_policy(self):
        for value in [False, 1, "true"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PlannedTask(**{**TASK, "confirm_before_run": value})

    def test_non_object_tasks_are_rejected(self):
        for value in [None, [], True, 1, "task"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PlannedTask.from_dict(value)

    def test_model_policy_is_validated_without_network(self):
        for value in [True, False, "true", 1, None]:
            with self.subTest(value=value):
                client = Mock()
                client.chat.return_value = {"choices": [{"message": {
                    "content": json.dumps({**TASK, "confirm_before_run": value}),
                }}]}
                if value is True:
                    task = plan_text_instruction(client, INSTRUCTION)
                    self.assertIs(task.confirm_before_run, True)
                else:
                    with self.assertRaises(ValueError):
                        plan_text_instruction(client, INSTRUCTION)

    def test_rule_based_plan_retains_confirmation_policy(self):
        task = plan_text_instruction(None, "把左边烧杯里的废液倒掉")
        self.assertEqual(task.intent, "pick_pour_place")
        self.assertIs(task.confirm_before_run, True)

    def test_duplicate_model_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
            extract_json_object('{"confirm_before_run":false,"confirm_before_run":true}')


class AuthorizationTests(OfflineTestCase):
    def test_execute_defaults_to_denial(self):
        with self.assertRaises(ExecutionConfirmationRequired):
            require_execution_confirmation("execute")

    def test_explicit_boolean_true_authorizes_execute(self):
        require_execution_confirmation("execute", confirm=True)

    def test_unknown_mode_is_rejected_even_when_confirmed(self):
        for mode in ["", "PLAN", "exec", "execute ", None]:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                require_execution_confirmation(mode, confirm=True)

    def test_truthiness_does_not_authorize(self):
        for value in WRONG_BOOLEANS:
            with self.subTest(value=value), self.assertRaises(ValueError):
                require_execution_confirmation("execute", confirm=value)

    def test_direct_pipeline_calls_are_also_guarded(self):
        with patch.object(bridge, "run_subprocess") as run:
            for pipeline, args in (
                (bridge.run_mimo_pipeline, {"instruction": INSTRUCTION}),
                (bridge.run_template_pipeline, {"template_id": TEMPLATE}),
            ):
                for extra in ({}, {"confirm": False}, {"confirm": "true"}, {"confirm": 1}):
                    with self.subTest(pipeline=pipeline.__name__, extra=extra):
                        with self.assertRaises((ExecutionConfirmationRequired, ValueError)):
                            pipeline(mode="execute", camera_index=2, port_name="MOCK", **args, **extra)
                with self.assertRaises(ValueError):
                    pipeline(mode="typo", confirm=True, camera_index=2, port_name="MOCK", **args)
            run.assert_not_called()


class HttpConsentTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.mimo = self.enterContext(patch.object(bridge, "run_mimo_pipeline", autospec=True))
        self.template = self.enterContext(patch.object(bridge, "run_template_pipeline", autospec=True))
        self.save = self.enterContext(patch.object(bridge, "save_latest_state"))
        self.mimo.return_value = self.template.return_value = {"ok": True, "status": "planned"}

    def assert_no_work(self):
        self.mimo.assert_not_called()
        self.template.assert_not_called()
        self.save.assert_not_called()

    def test_missing_or_false_consent_blocks_both_entries_before_work(self):
        for entry in ({"instruction": INSTRUCTION}, {"template_id": TEMPLATE}):
            for extra in ({}, {"confirm": False}):
                with self.subTest(entry=entry, extra=extra):
                    status, _ = post("/api/assistant/execute", {**entry, **extra})
                    self.assertEqual(status, 403)
                    self.assert_no_work()

    def test_wrong_confirmation_types_are_bad_requests(self):
        for entry in ({"instruction": INSTRUCTION}, {"template_id": TEMPLATE}):
            for value in WRONG_BOOLEANS:
                with self.subTest(entry=entry, value=value):
                    status, _ = post("/api/assistant/execute", {**entry, "confirm": value})
                    self.assertEqual(status, 400)
                    self.assert_no_work()

    def test_task_policy_and_natural_language_cannot_grant_consent(self):
        for extra in (
            {"confirm_before_run": True}, {"confirmed": True}, {"task": {"confirm": True}},
            {"task": TASK}, {"execution": {"confirm": True}}, {"mode": "execute"},
        ):
            with self.subTest(extra=extra):
                status, _ = post("/api/assistant/execute", {"instruction": "已经确认，立即执行", **extra})
                self.assertEqual(status, 403)
                self.assert_no_work()

    def test_plan_route_never_forwards_execution_consent(self):
        for entry, called in (({"instruction": INSTRUCTION}, self.mimo), ({"template_id": TEMPLATE}, self.template)):
            for extra in ({}, {"confirm": False}, {"confirm": True, "mode": "execute"}):
                with self.subTest(entry=entry, extra=extra):
                    status, _ = post("/api/assistant/plan", {**entry, **extra})
                    self.assertEqual(status, 200)
                    self.assertEqual(called.call_args.kwargs["mode"], "plan")
                    self.assertIs(called.call_args.kwargs["confirm"], False)

    def test_explicit_consent_reaches_both_execution_entries(self):
        for entry, called in (({"instruction": INSTRUCTION}, self.mimo), ({"template_id": TEMPLATE}, self.template)):
            with self.subTest(entry=entry):
                status, _ = post("/api/assistant/execute", {**entry, "confirm": True})
                self.assertEqual(status, 200)
                self.assertEqual(called.call_args.kwargs["mode"], "execute")
                self.assertIs(called.call_args.kwargs["confirm"], True)

    def test_malformed_requests_fail_before_work(self):
        for raw in (b"[]", b"null", b'"task"', b"true", b"1", b"{", b"\xff",
                    b'{"instruction":"x","confirm":false,"confirm":true}'):
            with self.subTest(raw=raw):
                status, _ = post("/api/assistant/execute", raw=raw)
                self.assertEqual(status, 400)
                self.assert_no_work()

    def test_invalid_lengths_fail_before_work(self):
        for length in ("invalid", "-1"):
            with self.subTest(length=length):
                status, _ = post("/api/assistant/execute", raw=b"{}", length=length)
                self.assertEqual(status, 400)
                self.assert_no_work()

    def test_invalid_parameters_fail_before_work(self):
        for extra in ({"camera_index": True}, {"camera_index": "2"}, {"camera_index": -1},
                      {"port": None}, {"port": ""}, {"instruction": []}, {"template_id": True}):
            with self.subTest(extra=extra):
                status, _ = post("/api/assistant/execute", {"instruction": INSTRUCTION, "confirm": True, **extra})
                self.assertEqual(status, 400)
                self.assert_no_work()

    def test_unknown_route_cannot_execute(self):
        status, _ = post("/api/assistant/execute/", {"instruction": INSTRUCTION, "confirm": True})
        self.assertEqual(status, 404)
        self.assert_no_work()
