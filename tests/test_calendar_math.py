"""Stdlib-only tests for generator, gold solver, and grader.

Run from this directory:

    python3 -m pytest tests/test_calendar_math.py -q
    python3 tests/test_calendar_math.py
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import calendar_math as cm  # noqa: E402


class GoldSolverTests(unittest.TestCase):
    def test_leap_year_february(self) -> None:
        self.assertEqual(cm.gold_add_days("2024-02-28", 1), "2024-02-29")
        self.assertEqual(cm.gold_add_days("2023-02-28", 1), "2023-03-01")

    def test_century_years(self) -> None:
        # 1900 is not leap; 2000 is.
        self.assertEqual(cm.gold_add_days("1900-02-28", 1), "1900-03-01")
        self.assertEqual(cm.gold_add_days("2000-02-28", 1), "2000-02-29")

    def test_year_boundary(self) -> None:
        self.assertEqual(cm.gold_add_days("2023-12-31", 1), "2024-01-01")
        self.assertEqual(cm.gold_add_days("2024-01-01", -1), "2023-12-31")

    def test_days_between_midnights(self) -> None:
        self.assertEqual(cm.gold_days_between("2024-01-01", "2024-01-02"), "1")
        self.assertEqual(cm.gold_days_between("2024-01-01", "2024-01-01"), "0")
        self.assertEqual(cm.gold_days_between("2024-02-28", "2024-03-01"), "2")
        self.assertEqual(cm.gold_days_between("2024-03-01", "2024-02-28"), "-2")

    def test_weekday_known(self) -> None:
        # 2024-02-29 was a Thursday; 2000-01-01 was a Saturday.
        self.assertEqual(cm.gold_weekday("2024-02-29"), "Thursday")
        self.assertEqual(cm.gold_weekday("2000-01-01"), "Saturday")
        self.assertEqual(cm.gold_weekday("1900-01-01"), "Monday")

    def test_example_from_spec_fills_answer(self) -> None:
        row = cm.example_from_spec({"task": "add_days", "start": "2024-02-28", "n": 1})
        self.assertEqual(row["answer"], "2024-02-29")
        self.assertIn("2024-02-28", row["question"])


class ParserAndGraderTests(unittest.TestCase):
    def _msg(self, text: str):
        return [{"role": "assistant", "content": text}]

    def test_exact_match_with_think_tags(self) -> None:
        completion = self._msg(
            "<think>Feb 2024 is leap</think>\n<answer>2024-02-29</answer>"
        )
        scores = cm.grade(completion, "2024-02-29", {"task": "add_days"})
        self.assertEqual(scores["exact_match"], 1.0)
        self.assertEqual(scores["format"], 1.0)
        self.assertEqual(scores["partial_credit"], 0.0)
        self.assertAlmostEqual(scores["reward"], 1.2)

    def test_wrong_answer_no_tags(self) -> None:
        scores = cm.grade(self._msg("I think it is Friday"), "Thursday", {"task": "weekday"})
        self.assertEqual(scores["exact_match"], 0.0)
        self.assertEqual(scores["format"], 0.0)
        self.assertEqual(scores["reward"], 0.0)

    def test_format_only(self) -> None:
        scores = cm.grade(self._msg("<answer>Friday</answer>"), "Thursday", {"task": "weekday"})
        self.assertEqual(scores["exact_match"], 0.0)
        self.assertEqual(scores["format"], 1.0)
        self.assertAlmostEqual(scores["partial_credit"], 0.3)
        self.assertAlmostEqual(scores["reward"], 0.2 + 0.2 * 0.3)

    def test_days_between_off_by_one(self) -> None:
        scores = cm.grade(self._msg("<answer>3</answer>"), "2", {"task": "days_between"})
        self.assertEqual(scores["partial_credit"], 0.5)

    def test_add_days_off_by_one(self) -> None:
        scores = cm.grade(
            self._msg("<answer>2024-03-01</answer>"),
            "2024-02-29",
            {"task": "add_days"},
        )
        self.assertEqual(scores["partial_credit"], 0.5)

    def test_bare_string_completion(self) -> None:
        self.assertEqual(cm.exact_match_score("<answer>Monday</answer>", "Monday"), 1.0)

    def test_pydantic_style_message_objects(self) -> None:
        class Msg:
            def __init__(self) -> None:
                self.role = "assistant"
                self.content = "<answer>2024-02-29</answer>"

        scores = cm.grade([Msg()], "2024-02-29", {"task_type": "add_days"})
        self.assertEqual(scores["exact_match"], 1.0)
        self.assertEqual(scores["format"], 1.0)
        self.assertAlmostEqual(scores["reward"], 1.2)

    def test_parse_strips_whitespace(self) -> None:
        self.assertEqual(cm.parse_answer("<answer>\n  42 \n</answer>"), "42")


class DatasetTests(unittest.TestCase):
    def test_train_size_and_uniqueness(self) -> None:
        rows = cm.build_rows(n=80, seed=0)
        self.assertEqual(len(rows), 80)
        questions = [r["question"] for r in rows]
        self.assertEqual(len(questions), len(set(questions)))
        for row in rows:
            self.assertEqual(row["answer"], cm.gold_answer(row))
            self.assertIn(row["task_type"], cm.TASK_TYPES)
            self.assertNotIn("task", row)
            self.assertNotIn("task", row["info"])
            self.assertTrue(row["question"])

    def test_eval_includes_edge_cases(self) -> None:
        rows = cm.build_rows(n=20, seed=1, include_edge_cases=True)
        questions = " ".join(r["question"] for r in rows)
        self.assertIn("1900-02-28", questions)
        self.assertIn("2024-02-29", questions)

    def test_pin_task(self) -> None:
        rows = cm.build_rows(n=15, seed=2, task="weekday")
        self.assertTrue(all(r["task_type"] == "weekday" for r in rows))

    def test_gold_dates_are_real(self) -> None:
        rows = cm.build_rows(n=60, seed=3, task="add_days")
        for row in rows:
            date.fromisoformat(row["answer"])  # raises on garbage


if __name__ == "__main__":
    unittest.main()
