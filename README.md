# calendar-math

Single-turn **calendar arithmetic** for RLVR / evals on the [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments).

Source: [github.com/devtechedge/calendar-math](https://github.com/devtechedge/calendar-math) · Hub: [devtechedge/calendar-math](https://app.primeintellect.ai/dashboard/environments/devtechedge/calendar-math)

The model is given one of three question types, reasons, and puts a final answer in `<answer>` tags. The grader is pure `datetime` — no LLM-as-judge, no fuzzy string matching on the main reward.

| Task | Example prompt | Gold answer |
| --- | --- | --- |
| `add_days` | What date is 1 day after 2024-02-28? | `2024-02-29` |
| `days_between` | How many days after 2024-02-28 is 2024-03-01? | `2` |
| `weekday` | What day of the week is 2024-02-29? | `Thursday` |

This is intentionally **not** reverse-text or word-count. Calendar reasoning is a documented LLM failure mode (leap years, century years, month lengths, weekday). The environment turns that into a dense, automatically-graded RL signal.

## Why this design

- **Verifiable.** Gold answers are produced by Python `datetime.date`. The same functions are the reference solver.
- **Hard where it matters.** Eval always includes curated edge cases: 1900-02-28 (century, not leap), 2000-02-28 (century, leap), 2024-02-29, year boundaries.
- **Not gameable by format alone.** Format is a 0.2 bonus. Exact match is the 1.0 term.
- **Shaping, not noise.** Off-by-one dates / day-counts score 0.5 partial credit — models routinely confuse inclusive vs exclusive counting. Adjacent weekdays score 0.3.
- **Configurable.** `num_train_examples`, `num_eval_examples`, `seed`, and an optional `task` pin.

## Reward

```
reward = 1.0 * exact_match + 0.2 * format + 0.2 * partial_credit
```

| Term | 1.0 when | Notes |
| --- | --- | --- |
| `exact_match` | parsed `<answer>` equals gold | dates ISO, weekdays canonical English, counts decimal integers |
| `format` | `<answer>...</answer>` present | extra prose outside the tags is ignored |
| `partial_credit` | near-miss as above | 0 when exact match already fired, so a perfect answer is **1.2** not 1.4 |

## Eval

`vf-eval` on the 15 curated edge cases (`num_eval_examples=15`, 1 rollout):

| Policy | avg reward | exact | format | notes |
| --- | --- | --- | --- | --- |
| Gold datetime solver (ceiling) | **1.200** | 1.000 | 1.000 | harness check |
| Naive calendar (`year%4` leaps, inclusive counts) | **0.768** | 0.533 | 1.000 | discrimination check |
| `minimax/minimax-m2.7:free` via OpenRouter | **0.880** | 0.733 | 0.733 | T=0, max_tokens=1536 |

The gold policy is a harness check: install, `load_environment`, rollouts, and the rubric all fire. The naive policy is a discrimination check: century non-leaps and inclusive day-counts do not rubber-stamp 1.2.

OpenRouter run (2026-08-25): 11/15 exact. The 4 misses were **truncated** mid-reasoning (365-day leap offsets and two `days_between` items) — format never closed, so exact=0. Century leap items (1900-02-28, 2000-02-28) scored 1.2.

```bash
uv run vf-eval calendar-math -n 15 -r 1 -p openrouter \
  -m minimax/minimax-m2.7:free --max-tokens 1536
```

## Installation

```bash
uv pip install -e .
python -m pytest tests/test_calendar_math.py -q
```

From the Hub:

```bash
prime env install devtechedge/calendar-math
```

```python
import verifiers as vf

env = vf.load_environment("calendar-math")
```

## `load_environment` arguments

| Arg | Default | Meaning |
| --- | --- | --- |
| `num_train_examples` | `500` | train split size |
| `num_eval_examples` | `100` | eval split size (edge cases prepended) |
| `seed` | `42` | train RNG; eval uses `seed + 1` |
| `task` | `None` | `"add_days"` \| `"days_between"` \| `"weekday"` \| mixed |

```bash
uv run vf-eval calendar-math -n 20
uv run vf-eval calendar-math -a '{"task": "weekday", "num_eval_examples": 40}'
```

Requires `verifiers>=0.1.14`. Dataset rows use `task_type` (not `task`): current verifiers treat `info["task"]` as a nested rollout payload.

## Gold solution

Dataset construction **is** the gold solver. For a row `info`:

```python
from datetime import date, timedelta

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]

def gold(info):
    if info["task_type"] == "add_days":
        return (date.fromisoformat(info["start"]) + timedelta(days=info["n"])).isoformat()
    if info["task_type"] == "days_between":
        a = date.fromisoformat(info["start"])
        b = date.fromisoformat(info["end"])
        return str((b - a).days)          # midnights that pass; same day = 0
    return WEEKDAYS[date.fromisoformat(info["start"]).weekday()]
```

`tests/test_calendar_math.py` asserts the solver against a hand-checked fixture list (leap years, century years, negative offsets, same-day diffs).

## Files

```
calendar_math.py              # generator, gold, grader, load_environment
pyproject.toml
README.md
LICENSE
tests/test_calendar_math.py
```

## What this is not

- Not a wrap of GSM8K or any public dataset.
- Not LLM-judged.
- Not multi-turn / tool-using. Those are the right shape for a **second** environment (meeting-conflict scheduler, timezone conversion with a tz database, etc.).

## License

MIT
