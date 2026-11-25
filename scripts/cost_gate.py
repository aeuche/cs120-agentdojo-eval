"""Measure dollars per attacked trace and freeze the eval sample if the grid exceeds $40.

Prices are OpenAI's published gpt-4o-mini rates, checked 2026-11-14: $0.15 per
million input tokens, $0.075 per million cached input tokens, $0.60 per million
output tokens. Cached tokens are the subset of prompt tokens reported in
``prompt_tokens_details.cached_tokens``.

The attacked grid is 560 workspace pairs, 4 defense conditions, and 3 repeats:
6,720 runs. The gate freezes a 200-pair subsample when that grid would cost
about $40 or more. Pair selection uses the suite's task ids and a fixed seed.
It does not read utility or security.

The subsample gives every user task the same 5 injection tasks, so AgentDojo's
cartesian product is exactly those 200 pairs. Ten dev pairs, on injection tasks
outside that set, are written to ``dev_pairs.json`` and kept out of
``eval_pairs.json``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

INPUT_PER_TOKEN = 0.15 / 1_000_000
CACHED_PER_TOKEN = 0.075 / 1_000_000
OUTPUT_PER_TOKEN = 0.60 / 1_000_000
PRICED_MODEL = "gpt-4o-mini-2024-07-18"
PAIRS = 560
CONDITIONS = 4
REPEATS = 3
ATTACKED_GRID_RUNS = PAIRS * CONDITIONS * REPEATS
BUDGET_DOLLARS = 40.0
SEED = 20261114
EVAL_PER_USER = 5
DEV_PAIRS = 10
FIXED_ATTACK = "important_instructions"
DEFENSES = (
    "spotlighting_with_delimiting",
    "transformers_pi_detector",
    "repeat_user_prompt",
    "tool_filter",
)


def task_number(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[1])


def call_dollars(call: dict) -> float | None:
    model = str(call.get("model") or "")
    if PRICED_MODEL not in model and not model.startswith("gpt-4o-mini"):
        return None
    prompt = int(call["prompt_tokens"])
    cached = min(int(call.get("cached_tokens") or 0), prompt)
    completion = int(call["completion_tokens"])
    return (prompt - cached) * INPUT_PER_TOKEN + cached * CACHED_PER_TOKEN + completion * OUTPUT_PER_TOKEN


def split_defense(pipeline_name: str) -> tuple[str, str]:
    for defense in DEFENSES:
        suffix = f"-{defense}"
        if pipeline_name.endswith(suffix):
            return pipeline_name[: -len(suffix)], defense
    return pipeline_name, "none"


def load_traces(runs: Path) -> list[dict]:
    traces = []
    if not runs.is_dir():
        return traces
    for path in sorted(runs.rglob("*.json")):
        if path.name == "condition.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "pipeline_name" not in payload:
            continue
        if "user_task_id" not in payload:
            continue
        usage = payload.get("api_usage") or {}
        calls = usage.get("calls") or []
        model, defense = split_defense(payload["pipeline_name"])
        attack = payload.get("attack_type") or "benign"
        priced = []
        unpriced = []
        for call in calls:
            dollars = call_dollars(call)
            if dollars is None:
                unpriced.append(call)
            else:
                priced.append((call, dollars))
        traces.append(
            {
                "path": str(path),
                "model": model,
                "defense": defense,
                "attack": attack,
                "user_task": payload["user_task_id"],
                "injection_task": payload.get("injection_task_id"),
                "has_utility": isinstance(payload.get("utility"), bool),
                "has_security": isinstance(payload.get("security"), bool),
                "utility": payload.get("utility") if isinstance(payload.get("utility"), bool) else None,
                "security": payload.get("security") if isinstance(payload.get("security"), bool) else None,
                "calls": len(calls),
                "prompt_tokens": sum(int(call["prompt_tokens"]) for call, _ in priced),
                "cached_tokens": sum(int(call.get("cached_tokens") or 0) for call, _ in priced),
                "completion_tokens": sum(int(call["completion_tokens"]) for call, _ in priced),
                "dollars": sum(dollars for _, dollars in priced),
                "unpriced_calls": len(unpriced),
            }
        )
    return traces


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_group(rows: list[dict]) -> dict:
    dollars = [row["dollars"] for row in rows]
    prompts = [row["prompt_tokens"] for row in rows]
    return {
        "n_traces": len(rows),
        "dollars_per_trace": _mean(dollars),
        "min_dollars": min(dollars) if dollars else None,
        "max_dollars": max(dollars) if dollars else None,
        "prompt_tokens_per_trace": _mean(prompts),
        "total_dollars": sum(dollars),
    }


def project(traces: list[dict]) -> dict:
    attacked = [
        row
        for row in traces
        if row["attack"] == FIXED_ATTACK
        and str(row["user_task"]).startswith("user_task_")
        and row["unpriced_calls"] == 0
        and row["calls"] > 0
    ]
    by_defense: dict[str, list[dict]] = {}
    for row in attacked:
        by_defense.setdefault(row["defense"], []).append(row)
    groups = {defense: summarize_group(rows) for defense, rows in sorted(by_defense.items())}
    none_mean = groups.get("none", {}).get("dollars_per_trace")
    tool_mean = groups.get("tool_filter", {}).get("dollars_per_trace")
    simple = None if none_mean is None else none_mean * ATTACKED_GRID_RUNS
    adjusted = None
    per_condition = None
    if none_mean is not None:
        per_condition = {
            "none": none_mean,
            "spotlighting_with_delimiting": none_mean,
            "transformers_pi_detector": none_mean,
            "tool_filter": tool_mean if tool_mean is not None else none_mean,
        }
        adjusted = PAIRS * REPEATS * sum(per_condition.values())
    candidates = [value for value in (simple, adjusted) if value is not None]
    projected = max(candidates) if candidates else None
    benign = [
        row
        for row in traces
        if row["attack"] == "benign"
        and str(row["user_task"]).startswith("user_task_")
        and row["defense"] == "none"
        and row["calls"] > 0
        and row["unpriced_calls"] == 0
    ]
    return {
        "pricing": {
            "model": PRICED_MODEL,
            "input_per_million": 0.15,
            "cached_input_per_million": 0.075,
            "output_per_million": 0.60,
            "source": "https://developers.openai.com/api/docs/models/gpt-4o-mini",
            "checked_on": "2026-11-14",
        },
        "attacked_grid_runs": ATTACKED_GRID_RUNS,
        "grid_formula": "560 pairs x 4 conditions x 3 repeats",
        "budget_dollars": BUDGET_DOLLARS,
        "dollars_per_attacked_trace": none_mean,
        "by_defense": groups,
        "per_condition_dollars_used_for_adjustment": per_condition,
        "projected_attacked_grid_dollars": simple,
        "projected_attacked_grid_dollars_adjusted": adjusted,
        "projection_used_for_gate": projected,
        "freeze": bool(projected is not None and projected >= BUDGET_DOLLARS),
        "benign_undefended": summarize_group(benign),
        "traces": [
            {
                "defense": row["defense"],
                "attack": row["attack"],
                "user_task": row["user_task"],
                "injection_task": row["injection_task"],
                "calls": row["calls"],
                "prompt_tokens": row["prompt_tokens"],
                "cached_tokens": row["cached_tokens"],
                "completion_tokens": row["completion_tokens"],
                "dollars": row["dollars"],
                "has_utility": row["has_utility"],
                "has_security": row["has_security"],
            }
            for row in traces
            if row["attack"] in {FIXED_ATTACK, "benign"}
        ],
    }


def build_sample(user_tasks: list[str], injection_tasks: list[str]) -> tuple[list[dict], list[dict], dict]:
    users = sorted(user_tasks, key=task_number)
    injections = sorted(injection_tasks, key=task_number)
    if len(users) != 40 or len(injections) != 14:
        raise SystemExit(f"Expected 40 user tasks and 14 injection tasks, got {len(users)} and {len(injections)}")
    rng = random.Random(SEED)
    order = injections.copy()
    rng.shuffle(order)
    eval_injections = order[:EVAL_PER_USER]
    held_out = order[EVAL_PER_USER:]
    eval_pairs = [
        {"user_task": user, "injection_task": injection}
        for user in users
        for injection in sorted(eval_injections, key=task_number)
    ]
    dev_users = users[::4]
    if len(dev_users) != DEV_PAIRS:
        raise SystemExit(f"Expected {DEV_PAIRS} dev user tasks, got {len(dev_users)}")
    dev_pairs = [
        {"user_task": user, "injection_task": held_out[index % len(held_out)]}
        for index, user in enumerate(dev_users)
    ]
    meta = {
        "suite": "workspace",
        "benchmark_version": "v1.2.2",
        "seed": SEED,
        "stratification": "Each of the 40 user tasks is crossed with the same 5 injection tasks.",
        "injection_tasks": sorted(eval_injections, key=task_number),
        "selection": "Task ids and seed only. Utility and security were not used.",
    }
    return eval_pairs, dev_pairs, meta


def smoke_flags(traces: list[dict]) -> dict:
    """Confirm the smoke tasks have both flags. This does not choose pairs."""
    expected = (
        ("none", "benign", None),
        ("none", FIXED_ATTACK, "injection_task_0"),
        ("tool_filter", FIXED_ATTACK, "injection_task_0"),
        ("transformers_pi_detector", FIXED_ATTACK, "injection_task_0"),
    )
    rows = []
    missing = []
    for defense, attack, injection in expected:
        for user in ("user_task_0", "user_task_1"):
            matches = [
                row
                for row in traces
                if row["defense"] == defense
                and row["attack"] == attack
                and row["user_task"] == user
                and (injection is None or row["injection_task"] == injection)
            ]
            ok = bool(matches) and all(row["has_utility"] and row["has_security"] for row in matches)
            rows.append(
                {
                    "defense": defense,
                    "attack": attack,
                    "user_task": user,
                    "injection_task": injection,
                    "n": len(matches),
                    "flags_present": ok,
                }
            )
            if not ok:
                missing.append(f"{defense} {attack} {user} {injection or 'benign'}")
    return {"rows": rows, "missing": missing, "ok": not missing}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_pairs(root: Path, eval_pairs: list[dict], dev_pairs: list[dict], meta: dict, projection: dict) -> None:
    eval_path = root / "eval_pairs.json"
    dev_path = root / "dev_pairs.json"
    eval_payload = {
        **meta,
        "dev_pairs_file": "dev_pairs.json",
        "reason": (
            "The 4x3 attacked grid projects to "
            f"${projection['projection_used_for_gate']:.2f}, which is at or above ${BUDGET_DOLLARS:.0f}."
        ),
        "dollars_per_attacked_trace": projection["dollars_per_attacked_trace"],
        "projected_attacked_grid_dollars": projection["projection_used_for_gate"],
        "n_pairs": len(eval_pairs),
        "pairs": eval_pairs,
    }
    dev_payload = {
        "suite": meta["suite"],
        "benchmark_version": meta["benchmark_version"],
        "seed": meta["seed"],
        "held_out_of": "eval_pairs.json",
        "selection": meta["selection"],
        "n_pairs": len(dev_pairs),
        "pairs": dev_pairs,
    }
    for path, payload in ((eval_path, eval_payload), (dev_path, dev_payload)):
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            current_pairs = current.get("pairs")
            if current_pairs != payload["pairs"]:
                raise SystemExit(f"{path} already exists with a different pair list. Leaving it unchanged.")
        write_json(path, payload)


def format_money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:.6f}" if value < 1 else f"${value:.2f}"


def self_check() -> None:
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite("v1.2.2", "workspace")
    eval_pairs, dev_pairs, meta = build_sample(list(suite.user_tasks), list(suite.injection_tasks))
    if len(eval_pairs) != 200 or len(dev_pairs) != 10:
        raise AssertionError((len(eval_pairs), len(dev_pairs)))
    eval_set = {(row["user_task"], row["injection_task"]) for row in eval_pairs}
    dev_set = {(row["user_task"], row["injection_task"]) for row in dev_pairs}
    if eval_set & dev_set:
        raise AssertionError("dev pairs overlap the eval sample")
    counts: dict[str, int] = {}
    injections = set()
    for row in eval_pairs:
        counts[row["user_task"]] = counts.get(row["user_task"], 0) + 1
        injections.add(row["injection_task"])
    if set(counts) != set(suite.user_tasks) or any(count != 5 for count in counts.values()):
        raise AssertionError(counts)
    if injections != set(meta["injection_tasks"]):
        raise AssertionError(injections)
    sample_call = {"model": PRICED_MODEL, "prompt_tokens": 1_000_000, "cached_tokens": 200_000, "completion_tokens": 1_000_000}
    dollars = call_dollars(sample_call)
    expected = 800_000 * INPUT_PER_TOKEN + 200_000 * CACHED_PER_TOKEN + 1_000_000 * OUTPUT_PER_TOKEN
    if dollars is None or abs(dollars - expected) > 1e-9:
        raise AssertionError((dollars, expected))
    print("cost_gate.py self-check passed")
    print(f"eval injection tasks: {', '.join(meta['injection_tasks'])}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("results/cost_gate.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--require-smoke",
        action="store_true",
        help="Exit non-zero unless tasks 0 and 1 have utility and security for the smoke conditions.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite("v1.2.2", "workspace")
    eval_pairs, dev_pairs, meta = build_sample(list(suite.user_tasks), list(suite.injection_tasks))
    traces = load_traces(args.runs)
    projection = project(traces)
    if projection["freeze"]:
        write_pairs(args.root, eval_pairs, dev_pairs, meta, projection)
        projection["eval_pairs"] = "eval_pairs.json"
        projection["dev_pairs"] = "dev_pairs.json"
        projection["n_eval_pairs"] = len(eval_pairs)
        projection["n_dev_pairs"] = len(dev_pairs)
    else:
        projection["eval_pairs"] = None
        projection["n_eval_pairs"] = None
    flags = smoke_flags(traces)
    projection["smoke_flags"] = flags
    write_json(args.out, projection)
    none = projection["dollars_per_attacked_trace"]
    used = projection["projection_used_for_gate"]
    print(f"dollars per undefended attacked trace: {format_money(none)}")
    print(f"projected 4x3 attacked grid: {format_money(used)}")
    if used is None:
        print("decision: no metered attacked traces yet, so the pair list stays unfrozen")
    elif projection["freeze"]:
        print("decision: freeze a 200-pair subsample")
    else:
        print("decision: full 560-pair grid stays under the $40 gate")
    if projection["freeze"]:
        print(f"wrote eval_pairs.json ({len(eval_pairs)} pairs) and dev_pairs.json ({len(dev_pairs)} pairs)")
    print(f"wrote {args.out}")
    if args.require_smoke and not flags["ok"]:
        print("smoke flags missing: " + "; ".join(flags["missing"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
