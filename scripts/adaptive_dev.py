"""Search the four adaptive variants on the held-out dev slice.

The slice is the 10-pair draw from ``scripts/cost_gate.py`` (seed 20261114, task
ids only). ``eval_pairs.json`` stays unwritten: the cost gate left confirmation
on all 560 workspace pairs. This script does not read utility or security to
choose pairs.

Each variant is run once, on those 10 pairs, against no defense and against
``spotlighting_with_delimiting``, ``transformers_pi_detector``, and
``tool_filter``. Every result is appended with
``attacks.adaptive.log_variant_outcome``, including variants that miss and
runs that error. Traces go to ``runs/adaptive_search`` (a cache symlink, same
layout as ``runs/study``). The log is ``runs/adaptive_dev.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cost_gate import build_sample

DEV_PAIRS_NAME = "dev_pairs.json"
DEV_LOG = ROOT / "runs" / "adaptive_dev.jsonl"
TRACE_CACHE = Path.home() / "Library" / "Caches" / "cs120-runs" / "adaptive_search"
TRACE_LINK = ROOT / "runs" / "adaptive_search"
MODEL = "GPT_4O_MINI_2024_07_18"
BENCHMARK = "v1.2.2"
SUITE = "workspace"
DEFENSES: tuple[str | None, ...] = (
    None,
    "spotlighting_with_delimiting",
    "transformers_pi_detector",
    "tool_filter",
)


def defense_label(defense: str | None) -> str:
    return defense or "none"


def dev_slice_name(defense: str | None) -> str:
    return f"{DEV_PAIRS_NAME}:{defense_label(defense)}"


def workspace_task_ids() -> tuple[list[str], list[str]]:
    users = [f"user_task_{index}" for index in range(40)]
    injections = [f"injection_task_{index}" for index in range(14)]
    return users, injections


def held_out_pairs() -> tuple[list[dict], list[dict], dict]:
    users, injections = workspace_task_ids()
    return build_sample(users, injections)


def write_dev_pairs(dev_pairs: list[dict], meta: dict) -> Path:
    path = ROOT / DEV_PAIRS_NAME
    payload = {
        "suite": SUITE,
        "benchmark_version": BENCHMARK,
        "seed": meta["seed"],
        "n_pairs": len(dev_pairs),
        "selection": (
            "Task ids and seed only, the same draw as scripts/cost_gate.py. "
            "Utility and security were not used. eval_pairs.json stays unwritten "
            "because the cost gate left confirmation on all 560 pairs. "
            "These pairs are only the adaptive search slice."
        ),
        "pairs": dev_pairs,
    }
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        current_pairs = current.get("pairs")
        if current_pairs != dev_pairs:
            raise SystemExit(f"{path} already exists with a different pair list. Leaving it unchanged.")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def prepare_trace_dir() -> Path:
    TRACE_CACHE.mkdir(parents=True, exist_ok=True)
    TRACE_LINK.parent.mkdir(parents=True, exist_ok=True)
    if TRACE_LINK.is_symlink():
        if TRACE_LINK.resolve() != TRACE_CACHE.resolve():
            raise SystemExit(f"{TRACE_LINK} points at {TRACE_LINK.resolve()}, not {TRACE_CACHE}")
    elif TRACE_LINK.exists():
        raise SystemExit(f"{TRACE_LINK} exists and is not the adaptive-search cache symlink.")
    else:
        TRACE_LINK.symlink_to(TRACE_CACHE)
    return TRACE_CACHE


def _transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    needles = ("rate limit", "ratelimit", "429", "timeout", "timed out", "overloaded", "connection", "502", "503", "504")
    return any(needle in text for needle in needles)


def _run_pairs(defense: str | None, variant: str, pairs: list[dict], logdir: str) -> list[dict]:
    import os

    os.chdir(ROOT)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import attacks.adaptive  # noqa: F401  # registers the four variants
    import defenses.study  # noqa: F401  # detector abort and local tool filter

    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import run_task_with_injection_tasks
    from agentdojo.logging import OutputLogger
    from agentdojo.models import ModelsEnum
    from agentdojo.task_suite.load_suites import get_suite

    if variant not in attacks.adaptive.VARIANTS:
        raise SystemExit(f"{variant} is not one of {attacks.adaptive.VARIANTS}")

    suite = get_suite(BENCHMARK, SUITE)
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=ModelsEnum[MODEL],
            model_id=None,
            defense=defense,
            system_message_name=None,
            system_message=None,
        )
    )
    attack = load_attack(variant, suite, pipeline)
    outcomes = []
    # TraceLogger reads logdir from the active OutputLogger. The benchmark CLI
    # enters that logger; calling the task runner directly has to do the same.
    with OutputLogger(logdir):
        for pair in pairs:
            user_task = pair["user_task"]
            injection_task = pair["injection_task"]
            record = {
                "user_task": user_task,
                "injection_task": injection_task,
                "defense": defense_label(defense),
                "model": "gpt-4o-mini-2024-07-18",
            }
            last_error: BaseException | None = None
            for attempt in range(3):
                try:
                    utility, security = run_task_with_injection_tasks(
                        suite,
                        pipeline,
                        suite.get_user_task_by_id(user_task),
                        attack,
                        Path(logdir),
                        False,
                        [injection_task],
                        BENCHMARK,
                    )
                    record["utility"] = bool(utility[(user_task, injection_task)])
                    record["security"] = bool(security[(user_task, injection_task)])
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 2 and _transient(exc):
                        time.sleep(5 * (attempt + 1))
                        continue
                    break
            if last_error is not None:
                record["utility"] = False
                record["security"] = False
                record["error"] = f"{type(last_error).__name__}: {last_error}"
            print(
                f"{defense_label(defense)} {variant} {user_task} {injection_task} "
                f"utility={record['utility']} security={record['security']}"
                + (f" error={record['error']}" if record.get("error") else ""),
                flush=True,
            )
            outcomes.append(record)
    return outcomes


def search_condition(defense: str | None, variant: str, pairs: list[dict], logdir: str) -> dict:
    """Run one variant on one defense. Failures become pair outcomes and are returned."""
    label = defense_label(defense)
    try:
        outcomes = _run_pairs(defense, variant, pairs, logdir)
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(f"{label} {variant} failed before pair results: {error}", flush=True)
        outcomes = [
            {
                "user_task": pair["user_task"],
                "injection_task": pair["injection_task"],
                "defense": label,
                "model": "gpt-4o-mini-2024-07-18",
                "utility": False,
                "security": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            for pair in pairs
        ]
    return {
        "variant": variant,
        "defense": label,
        "dev_slice": dev_slice_name(defense),
        "pair_outcomes": outcomes,
        "error": error,
    }


def _kept_log_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if str(row.get("dev_slice", "")).startswith(f"{DEV_PAIRS_NAME}:"):
            continue
        kept.append(line)
    return kept


def write_outcomes(results: list[dict]) -> None:
    """Rewrite this slice's lines, then append each one through the study logger."""
    from attacks.adaptive import log_variant_outcome

    kept = _kept_log_lines(DEV_LOG)
    if kept:
        DEV_LOG.parent.mkdir(parents=True, exist_ok=True)
        DEV_LOG.write_text("\n".join(kept) + "\n", encoding="utf-8")
    elif DEV_LOG.exists():
        DEV_LOG.unlink()
    for result in results:
        log_variant_outcome(
            result["variant"],
            dev_slice=result["dev_slice"],
            pair_outcomes=result["pair_outcomes"],
            path=DEV_LOG,
        )


def self_check() -> None:
    import tempfile

    from attacks.adaptive import VARIANTS, log_variant_outcome

    eval_pairs, dev_pairs, meta = held_out_pairs()
    if len(dev_pairs) != 10 or len(eval_pairs) != 200:
        raise AssertionError((len(eval_pairs), len(dev_pairs)))
    if {(row["user_task"], row["injection_task"]) for row in eval_pairs} & {
        (row["user_task"], row["injection_task"]) for row in dev_pairs
    }:
        raise AssertionError("dev pairs overlap the cost-gate eval draw")
    if meta["injection_tasks"] != ["injection_task_4", "injection_task_5", "injection_task_6", "injection_task_10", "injection_task_13"]:
        raise AssertionError(meta["injection_tasks"])
    if set(VARIANTS) != {
        "adaptive_paraphrase",
        "adaptive_delimiter_spoof",
        "adaptive_split",
        "adaptive_system_message",
    }:
        raise AssertionError(VARIANTS)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "adaptive_dev.jsonl"
        failed = log_variant_outcome(
            "adaptive_paraphrase",
            dev_slice="dev_pairs.json:none",
            pair_outcomes=[{"utility": False, "security": False}],
            path=path,
        )
        if failed["targeted_asr"] != 0.0 or failed["n_pairs"] != 1:
            raise AssertionError(failed)
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["variant"] != "adaptive_paraphrase" or row["targeted_asr"] != 0.0:
            raise AssertionError(row)
    print("adaptive_dev.py self-check passed")
    print(f"dev pairs: {len(dev_pairs)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true", help="Check the slice and the failure log, then exit.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel variant/defense runs.")
    parser.add_argument("--dry-run", action="store_true", help="Write the dev slice and print the jobs, then exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    from attacks.adaptive import VARIANTS

    eval_pairs, dev_pairs, meta = held_out_pairs()
    path = write_dev_pairs(dev_pairs, meta)
    print(f"wrote {path} ({len(dev_pairs)} pairs); eval draw stays unfrozen ({len(eval_pairs)} pairs)")
    jobs = [(defense, variant) for defense in DEFENSES for variant in VARIANTS]
    if args.dry_run:
        for defense, variant in jobs:
            print(f"{defense_label(defense)} {variant} {dev_slice_name(defense)} n={len(dev_pairs)}")
        return 0

    logdir = prepare_trace_dir()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(search_condition, defense, variant, dev_pairs, str(logdir)): (defense, variant)
            for defense, variant in jobs
        }
        for future in as_completed(futures):
            defense, variant = futures[future]
            result = future.result()
            results.append(result)
            successes = sum(bool(item["security"]) for item in result["pair_outcomes"])
            print(
                f"logged {result['dev_slice']} {variant}: {successes}/{len(result['pair_outcomes'])} security",
                flush=True,
            )
            write_outcomes(results)

    results.sort(key=lambda item: (item["defense"], item["variant"]))
    print("defense  variant  targeted_asr  utility  errors")
    for result in results:
        rows = result["pair_outcomes"]
        n = len(rows)
        asr = sum(bool(item["security"]) for item in rows) / n if n else None
        utility = sum(bool(item["utility"]) for item in rows) / n if n else None
        errors = sum(1 for item in rows if item.get("error"))
        asr_text = "—" if asr is None else f"{asr:.1%}"
        utility_text = "—" if utility is None else f"{utility:.1%}"
        print(f"{result['defense']}  {result['variant']}  {asr_text}  {utility_text}  {errors}")
    print(f"wrote {DEV_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
