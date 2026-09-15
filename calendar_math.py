"""calendar-math: single-turn calendar arithmetic with a deterministic grader.

The model solves one of three task types:

* ``add_days``     - offset a calendar date by N days
* ``days_between`` - count midnights between two dates (B − A)
* ``weekday``      - name the English weekday of a date

Answers must be wrapped in ``<answer>...</answer>``. The gold solver is
Python ``datetime.date`` - the same functions used to build the dataset.
Reward is exact match plus a small XML-format bonus and off-by-one
partial credit (models routinely confuse inclusive vs exclusive day
counts, and land one weekday away).
"""

from __future__ import annotations

import random
import re
from datetime import date, timedelta
from typing import Any, Literal, Sequence

TaskType = Literal["add_days", "days_between", "weekday"]
Difficulty = Literal["easy", "medium", "hard"]

WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

TASK_TYPES: tuple[TaskType, ...] = ("add_days", "days_between", "weekday")
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")

# Mix for randomly generated rows. Edge-case eval rows are injected separately.
DIFFICULTY_WEIGHTS = {"easy": 0.40, "medium": 0.35, "hard": 0.25}

SYSTEM_PROMPT = """You are solving calendar-arithmetic problems.

Rules:
- You may reason before answering.
- Put the final answer alone inside <answer>...</answer>.
- Dates must be ISO-8601: YYYY-MM-DD (zero-padded).
- Day counts are the number of midnights that pass from START to END.
  Example: 2024-01-01 to 2024-01-02 is 1, not 2. Same-day is 0.
- Weekdays are full English names: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.
- Do not put units, punctuation, or extra words inside <answer>.

Respond in this format:
<answer>
YOUR_ANSWER
</answer>
"""


# Curated eval rows that catch the failure modes this env exists to measure.
EDGE_CASES: tuple[dict[str, Any], ...] = (
    {"task": "add_days", "start": "2024-02-28", "n": 1, "difficulty": "hard"},
    {"task": "add_days", "start": "2023-02-28", "n": 1, "difficulty": "hard"},
    {"task": "add_days", "start": "1900-02-28", "n": 1, "difficulty": "hard"},
    {"task": "add_days", "start": "2000-02-28", "n": 1, "difficulty": "hard"},
    {"task": "add_days", "start": "2023-12-31", "n": 1, "difficulty": "hard"},
    {"task": "add_days", "start": "2024-01-01", "n": -1, "difficulty": "hard"},
    {"task": "add_days", "start": "2024-02-29", "n": 365, "difficulty": "hard"},
    {"task": "add_days", "start": "2024-02-29", "n": -365, "difficulty": "hard"},
    {"task": "weekday", "start": "2024-02-29", "difficulty": "hard"},
    {"task": "weekday", "start": "2000-01-01", "difficulty": "hard"},
    {"task": "weekday", "start": "1900-01-01", "difficulty": "hard"},
    {"task": "days_between", "start": "2024-02-28", "end": "2024-03-01", "difficulty": "hard"},
    {"task": "days_between", "start": "1999-12-31", "end": "2000-01-01", "difficulty": "hard"},
    {"task": "days_between", "start": "2024-01-01", "end": "2024-01-01", "difficulty": "easy"},
    {"task": "days_between", "start": "2024-02-29", "end": "2025-02-28", "difficulty": "hard"},
)


# ---------------------------------------------------------------------------
# Gold solver - this is the reference solution. Dataset answers are produced
# by these functions, so a correct implementation scores 1.0 by construction.
# ---------------------------------------------------------------------------


def parse_iso_date(value: str) -> date:
    match = ISO_DATE_RE.match(value.strip())
    if not match:
        raise ValueError(f"not an ISO date: {value!r}")
    year, month, day = (int(part) for part in match.groups())
    return date(year, month, day)


def gold_add_days(start: str, n: int) -> str:
    return (parse_iso_date(start) + timedelta(days=int(n))).isoformat()


def gold_days_between(start: str, end: str) -> str:
    return str((parse_iso_date(end) - parse_iso_date(start)).days)


def gold_weekday(start: str) -> str:
    return WEEKDAYS[parse_iso_date(start).weekday()]


def gold_answer(example: dict[str, Any]) -> str:
    """Return the canonical answer string for a generated example."""
    info = example.get("info", example)
    task: TaskType = info.get("task_type") or info["task"]
    if task == "add_days":
        return gold_add_days(info["start"], int(info["n"]))
    if task == "days_between":
        return gold_days_between(info["start"], info["end"])
    if task == "weekday":
        return gold_weekday(info["start"])
    raise ValueError(f"unknown task: {task}")


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def _rand_date(rng: random.Random, y0: int, y1: int) -> date:
    start = date(y0, 1, 1)
    end = date(y1, 12, 31)
    return start + timedelta(days=rng.randint(0, (end - start).days))


def _edge_start(rng: random.Random) -> date:
    """Bias hard items toward month/year/leap boundaries."""
    year = rng.choice([1900, 2000, 2023, 2024, 2025, 2100])
    choices = [
        date(year, 1, 1),
        date(year, 12, 31),
        date(year, 2, 28),
        date(year, 3, 1),
        date(year, 1, 31),
    ]
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        choices.append(date(year, 2, 29))
    return rng.choice(choices)


def _question_for(task: TaskType, info: dict[str, Any]) -> str:
    if task == "add_days":
        n = int(info["n"])
        start = info["start"]
        if n >= 0:
            return (
                f"What date is {n} day{'s' if n != 1 else ''} after {start}? "
                "Reply with a single YYYY-MM-DD date."
            )
        mag = abs(n)
        return (
            f"What date is {mag} day{'s' if mag != 1 else ''} before {start}? "
            "Reply with a single YYYY-MM-DD date."
        )
    if task == "days_between":
        return (
            f"How many days after {info['start']} is {info['end']}? "
            "Count the number of midnights that pass "
            "(so 2024-01-01 to 2024-01-02 is 1; the same day is 0). "
            "Reply with a single integer. The integer may be negative if the "
            "end date is earlier than the start date."
        )
    if task == "weekday":
        return (
            f"What day of the week is {info['start']}? "
            "Reply with the full English weekday name "
            "(Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, or Sunday)."
        )
    raise ValueError(f"unknown task: {task}")


def generate_example(
    rng: random.Random,
    *,
    task: TaskType | None = None,
    difficulty: Difficulty | None = None,
) -> dict[str, Any]:
    """Generate one fully-specified example with gold answer filled in."""
    difficulty = difficulty or rng.choices(
        population=list(DIFFICULTY_WEIGHTS),
        weights=list(DIFFICULTY_WEIGHTS.values()),
        k=1,
    )[0]
    task = task or rng.choice(TASK_TYPES)

    if difficulty == "easy":
        y0, y1 = 2018, 2026
    elif difficulty == "medium":
        y0, y1 = 1995, 2035
    else:
        y0, y1 = 1890, 2110

    if task == "add_days":
        start = _edge_start(rng) if difficulty == "hard" and rng.random() < 0.45 else _rand_date(rng, y0, y1)
        if difficulty == "easy":
            n = rng.randint(1, 7) * rng.choice((-1, 1))
        elif difficulty == "medium":
            n = rng.randint(8, 60) * rng.choice((-1, 1))
        else:
            n = rng.randint(60, 400) * rng.choice((-1, 1))
        info = {"task_type": task, "difficulty": difficulty, "start": start.isoformat(), "n": n}
    elif task == "days_between":
        a = _rand_date(rng, y0, y1)
        if difficulty == "easy":
            delta = rng.randint(0, 12)
        elif difficulty == "medium":
            delta = rng.randint(13, 90) * rng.choice((-1, 1))
        else:
            delta = rng.randint(90, 450) * rng.choice((-1, 1))
        b = a + timedelta(days=delta)
        info = {
            "task_type": task,
            "difficulty": difficulty,
            "start": a.isoformat(),
            "end": b.isoformat(),
        }
    else:
        start = _edge_start(rng) if difficulty == "hard" and rng.random() < 0.5 else _rand_date(rng, y0, y1)
        info = {"task_type": task, "difficulty": difficulty, "start": start.isoformat()}

    answer = gold_answer({"info": info})
    info["gold"] = answer
    return {
        "question": _question_for(task, info),
        "answer": answer,
        "task_type": task,
        "info": info,
    }


def example_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Materialize a hand-written spec (used for curated edge cases)."""
    task: TaskType = spec["task"]
    info = {k: v for k, v in spec.items() if k != "task"}
    info["task_type"] = task
    info.setdefault("difficulty", "hard")
    answer = gold_answer({"info": info})
    info["gold"] = answer
    return {
        "question": _question_for(task, info),
        "answer": answer,
        "task_type": task,
        "info": info,
    }


def build_rows(
    *,
    n: int,
    seed: int,
    include_edge_cases: bool = False,
    task: TaskType | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    if include_edge_cases:
        for spec in EDGE_CASES:
            if task is None or spec["task"] == task:
                rows.append(example_from_spec(spec))
    seen = {row["question"] for row in rows}
    guard = 0
    while len(rows) < n and guard < n * 20:
        guard += 1
        example = generate_example(rng, task=task)
        if example["question"] in seen:
            continue
        seen.add(example["question"])
        rows.append(example)
    if len(rows) < n:
        raise RuntimeError(f"could only generate {len(rows)} unique rows (wanted {n})")
    return rows[:n]


# ---------------------------------------------------------------------------
# Parser + rubric (stdlib - also used by tests without verifiers installed)
# ---------------------------------------------------------------------------


def extract_completion_text(completion: Any) -> str:
    """Normalize verifiers completion objects down to a single string."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for message in reversed(completion):
            role, content = _message_role_content(message)
            if role not in (None, "assistant"):
                continue
            text = _content_to_text(content)
            if text:
                return text
        return ""
    if isinstance(completion, dict):
        return _content_to_text(completion.get("content", ""))
    role, content = _message_role_content(completion)
    if content is not None:
        return _content_to_text(content)
    return str(completion)


def _message_role_content(message: Any) -> tuple[Any, Any]:
    if isinstance(message, dict):
        return message.get("role"), message.get("content")
    role = getattr(message, "role", None)
    content = getattr(message, "content", None)
    if role is None and content is None and hasattr(message, "model_dump"):
        dumped = message.model_dump()
        if isinstance(dumped, dict):
            return dumped.get("role"), dumped.get("content")
    return role, content


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)

def parse_answer(completion: Any) -> str | None:
    text = extract_completion_text(completion)
    match = ANSWER_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def has_answer_tags(completion: Any) -> bool:
    return parse_answer(completion) is not None


def normalize_answer(value: str) -> str:
    return " ".join(value.strip().split())


def exact_match_score(completion: Any, answer: str) -> float:
    parsed = parse_answer(completion)
    if parsed is None:
        return 0.0
    return 1.0 if normalize_answer(parsed) == normalize_answer(str(answer)) else 0.0


def format_score(completion: Any) -> float:
    return 1.0 if has_answer_tags(completion) else 0.0


def _try_parse_date(value: str) -> date | None:
    try:
        return parse_iso_date(value)
    except ValueError:
        return None


def partial_credit_score(completion: Any, answer: str, info: dict[str, Any] | None = None) -> float:
    """Shaping term. Exact match already scores 1.0 on the main reward.

    * add_days: 0.5 if the predicted date is exactly one day off
    * days_between: 0.5 if off by one (inclusive/exclusive confusion)
    * weekday: 0.3 if adjacent weekday (wraps around the week)
    """
    parsed = parse_answer(completion)
    if parsed is None:
        return 0.0
    parsed_norm = normalize_answer(parsed)
    gold = normalize_answer(str(answer))
    if parsed_norm == gold:
        return 0.0  # don't double-count a perfect answer

    task = (info or {}).get("task_type") or (info or {}).get("task")
    if task == "add_days" or _try_parse_date(gold) is not None:
        pred_d = _try_parse_date(parsed_norm)
        gold_d = _try_parse_date(gold)
        if pred_d is not None and gold_d is not None:
            return 0.5 if abs((pred_d - gold_d).days) == 1 else 0.0
    if task == "days_between":
        try:
            return 0.5 if abs(int(parsed_norm) - int(gold)) == 1 else 0.0
        except ValueError:
            return 0.0
    if task == "weekday" or gold in WEEKDAYS:
        if parsed_norm not in WEEKDAYS or gold not in WEEKDAYS:
            return 0.0
        pi = WEEKDAYS.index(parsed_norm)
        gi = WEEKDAYS.index(gold)
        return 0.3 if min((pi - gi) % 7, (gi - pi) % 7) == 1 else 0.0
    return 0.0


def grade(completion: Any, answer: str, info: dict[str, Any] | None = None) -> dict[str, float]:
    """Return the three rubric terms plus the weighted scalar used in RL."""
    exact = exact_match_score(completion, answer)
    fmt = format_score(completion)
    partial = partial_credit_score(completion, answer, info)
    weighted = 1.0 * exact + 0.2 * fmt + 0.2 * partial
    return {
        "exact_match": exact,
        "format": fmt,
        "partial_credit": partial,
        "reward": weighted,
    }


# ---------------------------------------------------------------------------
# load_environment - Hub / verifiers v0 entrypoint
# ---------------------------------------------------------------------------


def load_environment(
    num_train_examples: int = 500,
    num_eval_examples: int = 100,
    seed: int = 42,
    task: TaskType | None = None,
    **kwargs: Any,
) -> Any:
    """Build a ``vf.SingleTurnEnv`` for calendar arithmetic.

    Dataset rows use ``task_type`` (not ``task``). Verifiers ≥0.1 treats
    ``info["task"]`` as a nested rollout payload, so a string task name
    there crashes ``vf-eval``.

    Parameters
    ----------
    num_train_examples:
        Size of the train split. Default 500 - dense enough for a first
        GRPO run without being a toy set of 8 rows.
    num_eval_examples:
        Size of the eval split. Curated leap-year / century / year-boundary
        rows are always prepended, then unique random rows fill the rest.
    seed:
        Dataset RNG seed. Train uses ``seed``; eval uses ``seed + 1``.
    task:
        Pin generation to one task type, or ``None`` for a uniform mix.
    """
    import verifiers as vf
    from datasets import Dataset

    train_rows = build_rows(n=num_train_examples, seed=seed, task=task)
    eval_rows = build_rows(
        n=num_eval_examples,
        seed=seed + 1,
        include_edge_cases=True,
        task=task,
    )
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)

    parser = vf.XMLParser(["answer"], answer_field="answer")

    def exact_match_reward(completion, answer, **_kwargs) -> float:
        parsed = parser.parse_answer(completion)
        if parsed is None:
            return exact_match_score(completion, answer)
        return 1.0 if normalize_answer(parsed) == normalize_answer(str(answer)) else 0.0

    def format_reward(completion, **_kwargs) -> float:
        # Use our 0/1 tag check, not XMLParser.get_format_reward_func():
        # the library helper still awards 0.2 with no tags, which would let
        # format-free completions farm a constant bonus.
        return format_score(completion)

    def partial_credit_reward(completion, answer, info=None, **_kwargs) -> float:
        return partial_credit_score(completion, answer, info)

    rubric = vf.Rubric(
        funcs=[exact_match_reward, format_reward, partial_credit_reward],
        weights=[1.0, 0.2, 0.2],
        parser=parser,
    )


    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        **{k: v for k, v in kwargs.items() if k in ("max_concurrent",)},
    )
