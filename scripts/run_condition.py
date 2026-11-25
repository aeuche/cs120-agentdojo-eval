"""Run one AgentDojo condition, optionally restricted to a frozen pair list.

The study grid is workspace, benchmark v1.2.2, three repeats, temperature left at
AgentDojo's OpenAI default of 0. This wrapper only selects the suite, model,
defense, attack, log directory, and pairs. Scoring stays in ``analysis/score.py``.

Examples:

    python scripts/run_condition.py --model GPT_4O_MINI_2024_07_18 --repeats 1 \\
        --user-task user_task_0 --user-task user_task_1

    python scripts/run_condition.py --model GPT_4O_MINI_2024_07_18 \\
        --attack important_instructions --pairs eval_pairs.json --repeats 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPTIVE_MODULE = "attacks.adaptive"
STUDY_DEFENSES_MODULE = "defenses.study"

DEFENSES = (
    "spotlighting_with_delimiting",
    "transformers_pi_detector",
    "tool_filter",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="workspace", help="AgentDojo suite. The study uses workspace.")
    parser.add_argument(
        "--benchmark-version",
        default="v1.2.2",
        help="Benchmark version passed through to AgentDojo.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="ModelsEnum name, for example GPT_4O_MINI_2024_07_18 or LOCAL.",
    )
    parser.add_argument("--model-id", default=None, help="Served model id. Required in the study for LOCAL.")
    parser.add_argument(
        "--defense",
        default=None,
        choices=DEFENSES,
        help="Omit for the undefended agent.",
    )
    parser.add_argument(
        "--attack",
        default=None,
        help="Omit for a benign run. Fixed study attack: important_instructions.",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="Frozen pair list. Object with a 'pairs' array of {user_task, injection_task}.",
    )
    parser.add_argument("--user-task", action="append", default=[], help="User task id. Repeatable. Used when --pairs is omitted.")
    parser.add_argument(
        "--injection-task",
        action="append",
        default=[],
        help="Injection task id. Repeatable. Cartesian with --user-task when --pairs is omitted.",
    )
    parser.add_argument("--logdir", type=Path, default=Path("runs"), help="Root directory for traces.")
    parser.add_argument("--repeats", type=int, default=3, help="Separate log directories, so repeats do not overwrite.")
    parser.add_argument("--module-to-load", action="append", default=[], help="Extra modules to register before the run.")
    parser.add_argument("--force-rerun", action="store_true", help="Rerun tasks that already have a trace.")
    parser.add_argument("--dry-run", action="store_true", help="Print the benchmark commands and exit.")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.model in {"LOCAL", "local"} and not args.model_id:
        parser.error("--model-id is required for the local model")
    if args.pairs is not None and (args.user_task or args.injection_task):
        parser.error("Pass either --pairs or task filters, not both")
    return args


def load_pairs(path: Path, suite: str) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if payload.get("suite") not in (None, suite):
            raise SystemExit(f"{path} is for suite {payload.get('suite')!r}, not {suite!r}")
        rows = payload.get("pairs")
    else:
        rows = payload
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{path} does not contain a non-empty pair list")
    pairs = []
    for row in rows:
        try:
            pairs.append({"user_task": row["user_task"], "injection_task": row["injection_task"]})
        except (KeyError, TypeError) as exc:
            raise SystemExit(f"Pair entry {row!r} needs user_task and injection_task") from exc
    return pairs


def attacked_batches(pairs: list[dict[str, str]]) -> list[tuple[list[str], list[str]]]:
    """Group user tasks that share the same injection-task set.

    AgentDojo crosses every ``--user-task`` with every ``--injection-task``.
    Grouping keeps a stratified subsample from picking up pairs that were not frozen.
    """
    by_user: dict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        injections = by_user[pair["user_task"]]
        if pair["injection_task"] not in injections:
            injections.append(pair["injection_task"])
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for user_task, injections in by_user.items():
        groups[tuple(sorted(injections))].append(user_task)
    return [(users, list(injections)) for injections, users in groups.items()]


def benchmark_model_token(requested: str) -> str:
    """Translate an enum name to the token ``agentdojo.scripts.benchmark`` accepts."""
    from agentdojo.models import ModelsEnum
    from agentdojo.scripts import benchmark as benchmark_mod

    if requested in ModelsEnum.__members__:
        member = ModelsEnum[requested]
    else:
        try:
            member = ModelsEnum(requested)
        except ValueError as exc:
            names = ", ".join(ModelsEnum.__members__)
            raise SystemExit(f"Unknown model {requested!r}. Use an enum name: {names}") from exc

    model_param = next(param for param in benchmark_mod.main.params if param.name == "model")
    # Click matches the enum name (GPT_4O_MINI_2024_07_18). str(member) is the dashed id, which the CLI rejects.
    accepted = []
    for candidate in (member.name, str(member), member.value):
        try:
            model_param.type.convert(candidate, model_param, None)
        except Exception:
            continue
        accepted.append(candidate)
    if accepted:
        return accepted[0]
    names = ", ".join(choice.name for choice in model_param.type.choices)
    raise SystemExit(f"AgentDojo does not accept {requested!r}. Choices: {names}")


def condition_label(model: str, model_id: str | None, defense: str | None, attack: str | None) -> tuple[str, str, str]:
    model_label = model
    if model in {"LOCAL", "local"} and model_id:
        slug = model_id.replace("/", "_")
        model_label = f"local__{slug}"
    defense_label = defense or "none"
    attack_label = attack or "benign"
    return model_label, defense_label, attack_label


def build_command(
    *,
    python: str,
    suite: str,
    benchmark_version: str,
    model_token: str,
    model_id: str | None,
    defense: str | None,
    attack: str | None,
    logdir: Path,
    user_tasks: list[str],
    injection_tasks: list[str],
    modules: list[str],
    force_rerun: bool,
) -> list[str]:
    command = [
        python,
        "-m",
        "agentdojo.scripts.benchmark",
        "--suite",
        suite,
        "--benchmark-version",
        benchmark_version,
        "--model",
        model_token,
        "--logdir",
        str(logdir),
    ]
    for module in modules:
        command.extend(["--module-to-load", module])
    if model_id:
        command.extend(["--model-id", model_id])
    if defense:
        command.extend(["--defense", defense])
    if attack:
        command.extend(["--attack", attack])
    if force_rerun:
        command.append("--force-rerun")
    for user_task in user_tasks:
        command.extend(["--user-task", user_task])
    for injection_task in injection_tasks:
        command.extend(["--injection-task", injection_task])
    return command


def write_condition_meta(logdir: Path, payload: dict) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "condition.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    modules = list(dict.fromkeys([*args.module_to_load, STUDY_DEFENSES_MODULE, ADAPTIVE_MODULE]))
    model_token = benchmark_model_token(args.model)
    model_label, defense_label, attack_label = condition_label(args.model, args.model_id, args.defense, args.attack)

    if args.pairs is not None:
        pairs = load_pairs(args.pairs, args.suite)
        if args.attack is None:
            batches = [(sorted({pair["user_task"] for pair in pairs}), [])]
        else:
            batches = attacked_batches(pairs)
    else:
        batches = [(args.user_task, args.injection_task if args.attack else [])]

    exit_code = 0
    for repeat in range(1, args.repeats + 1):
        for batch_index, (user_tasks, injection_tasks) in enumerate(batches, start=1):
            logdir = ROOT / args.logdir / model_label / defense_label / attack_label / f"repeat_{repeat}"
            if len(batches) > 1:
                logdir = logdir / f"batch_{batch_index}"
            meta = {
                "suite": args.suite,
                "benchmark_version": args.benchmark_version,
                "model": args.model,
                "model_token": model_token,
                "model_id": args.model_id,
                "defense": args.defense,
                "attack": args.attack,
                "repeat": repeat,
                "pairs_file": str(args.pairs) if args.pairs else None,
                "user_tasks": user_tasks,
                "injection_tasks": injection_tasks,
            }
            command = build_command(
                python=sys.executable,
                suite=args.suite,
                benchmark_version=args.benchmark_version,
                model_token=model_token,
                model_id=args.model_id,
                defense=args.defense,
                attack=args.attack,
                logdir=logdir,
                user_tasks=user_tasks,
                injection_tasks=injection_tasks,
                modules=modules,
                force_rerun=args.force_rerun,
            )
            print(" ".join(command), flush=True)
            if args.dry_run:
                continue
            write_condition_meta(logdir, meta)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                return completed.returncode
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
