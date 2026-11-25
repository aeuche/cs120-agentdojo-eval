"""Confirm the best adaptive variant once per defense on all 560 workspace pairs.

The dev log is ``runs/adaptive_dev.jsonl``. For each defense, the confirmed
variant is the one with the highest targeted ASR on that defense's 10-pair
slice. A tie goes to the variant with the higher mean ASR across defenses,
then to the lexicographically later name (the same name order
``analysis/score.py`` uses when several adaptive attacks are present).

Traces land under ``runs/study`` so the fixed-attack grid and this confirmation
are scored together. One repeat. ``eval_pairs.json`` stays unwritten: the cost
gate left the sample at all 560 pairs. After the traces are complete, this
rewrites ``results/workspace_grid.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEV_LOG = ROOT / "runs" / "adaptive_dev.jsonl"
STUDY = ROOT / "runs" / "study"
CACHE = Path.home() / "Library" / "Caches" / "cs120-runs" / "study"
LOGS = Path.home() / "Library" / "Caches" / "cs120-confirm-logs"
STATUS = Path.home() / "Library" / "Caches" / "cs120-confirm-status.json"
HF_HOME = str(Path.home() / "Library" / "Caches" / "cs120-hf")
REPORT = ROOT / "results" / "workspace_grid.json"
SELECTION = ROOT / "results" / "adaptive_confirmation.json"
MODEL = "GPT_4O_MINI_2024_07_18"
MODEL_ID = "gpt-4o-mini-2024-07-18"
BENCHMARK = "v1.2.2"
SUITE = "workspace"
N_USERS = 40
N_INJECTIONS = 14
N_PAIRS = N_USERS * N_INJECTIONS
DEFENSES: tuple[str | None, ...] = (
    None,
    "spotlighting_with_delimiting",
    "transformers_pi_detector",
    "tool_filter",
)
DETECTOR = "transformers_pi_detector"
API_SLOTS = 4
DETECTOR_SLOTS = 1
MAX_ATTEMPTS = 5
MIN_FREE_PERCENT = 30
MIN_FREE_PERCENT_DETECTOR = 45
N_SHARDS = 8


def python_executable() -> str:
    candidates = [
        Path.home() / "Library" / "Caches" / "cs120-final-venv" / "bin" / "python",
        ROOT / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def defense_label(defense: str | None) -> str:
    return defense or "none"


def load_dev_rows(path: Path = DEV_LOG) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Missing dev log {path}. Run scripts/adaptive_dev.py before confirmation.")
    latest: dict[tuple[str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        dev_slice = str(row.get("dev_slice", ""))
        if not dev_slice.startswith("dev_pairs.json:"):
            continue
        defense = dev_slice.split(":", 1)[1]
        variant = row["variant"]
        asr = row.get("targeted_asr")
        if asr is None:
            raise SystemExit(f"{variant} on {defense} has no targeted ASR in {path}.")
        latest[(defense, variant)] = {
            "defense": defense,
            "variant": variant,
            "targeted_asr": float(asr),
            "n_pairs": int(row.get("n_pairs") or 0),
            "utility_under_attack": row.get("utility_under_attack"),
            "timestamp": row.get("timestamp"),
        }
    return list(latest.values())


def select_variants(rows: list[dict]) -> dict[str, dict]:
    """Pick one variant per defense. Higher dev ASR wins. Ties use mean ASR, then name."""
    by_defense: dict[str, list[dict]] = defaultdict(list)
    by_variant: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_defense[row["defense"]].append(row)
        by_variant[row["variant"]].append(row["targeted_asr"])
    mean_asr = {variant: sum(values) / len(values) for variant, values in by_variant.items()}
    selected = {}
    for defense in (defense_label(item) for item in DEFENSES):
        choices = by_defense.get(defense)
        if not choices:
            raise SystemExit(f"Dev log has no rows for {defense}.")
        winner = max(choices, key=lambda row: (row["targeted_asr"], mean_asr[row["variant"]], row["variant"]))
        selected[defense] = {
            "variant": winner["variant"],
            "dev_targeted_asr": winner["targeted_asr"],
            "dev_n_pairs": winner["n_pairs"],
            "dev_utility_under_attack": winner["utility_under_attack"],
            "tie_break_mean_asr": mean_asr[winner["variant"]],
        }
    return selected


def user_tasks() -> list[str]:
    return [f"user_task_{index}" for index in range(N_USERS)]


def shard_tasks(tasks: list[str], n_shards: int) -> list[list[str]]:
    groups: list[list[str]] = [[] for _ in range(n_shards)]
    for index, task in enumerate(tasks):
        groups[index % n_shards].append(task)
    return [group for group in groups if group]


def trace_complete(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("utility"), bool) and isinstance(payload.get("security"), bool)


def count_done(logdir: Path) -> int:
    if not logdir.exists():
        return 0
    done = 0
    for path in logdir.rglob("*.json"):
        if path.name == "condition.json":
            continue
        if not any(part.startswith("user_task_") for part in path.parts):
            continue
        if trace_complete(path):
            done += 1
    return done


class Job:
    def __init__(self, *, name: str, defense: str | None, attack: str, tasks: list[str], logdir: Path, kind: str) -> None:
        self.name = name
        self.defense = defense
        self.attack = attack
        self.tasks = tasks
        self.logdir = logdir
        self.kind = kind
        self.attempts = 0
        self.proc: subprocess.Popen | None = None
        self.log_handle = None
        self.not_before = 0.0

    @property
    def expected(self) -> int:
        return len(self.tasks) * N_INJECTIONS

    def complete(self) -> bool:
        return count_done(self.logdir) >= self.expected

    def command(self, python: str) -> list[str]:
        command = [
            python,
            str(ROOT / "scripts" / "run_condition.py"),
            "--suite",
            SUITE,
            "--benchmark-version",
            BENCHMARK,
            "--model",
            MODEL,
            "--repeats",
            "1",
            "--attack",
            self.attack,
            "--logdir",
            str(self.logdir),
        ]
        if self.defense:
            command.extend(["--defense", self.defense])
        for task in self.tasks:
            command.extend(["--user-task", task])
        return command


def make_jobs(selected: dict[str, dict]) -> list[Job]:
    jobs: list[Job] = []
    tasks = user_tasks()
    for defense in DEFENSES:
        label = defense_label(defense)
        attack = selected[label]["variant"]
        kind = "detector" if defense == DETECTOR else "api"
        for index, group in enumerate(shard_tasks(tasks, N_SHARDS)):
            logdir = CACHE / SUITE / label / attack / f"shard_{index}"
            jobs.append(
                Job(
                    name=f"confirm-{label}-{attack}-s{index}",
                    defense=defense,
                    attack=attack,
                    tasks=group,
                    logdir=logdir,
                    kind=kind,
                )
            )
    return jobs


def child_env() -> dict[str, str]:
    copied = os.environ.copy()
    copied["HF_HOME"] = HF_HOME
    copied["PYTHONUNBUFFERED"] = "1"
    copied["TOKENIZERS_PARALLELISM"] = "false"
    return copied


def free_percent() -> int:
    try:
        output = subprocess.check_output(["memory_pressure"], text=True, timeout=10)
    except (subprocess.SubprocessError, OSError, ValueError):
        return 100
    for line in output.splitlines():
        if "System-wide memory free percentage:" in line:
            return int(line.rsplit(":", 1)[1].strip().rstrip("%"))
    return 100


def live_commands() -> list[str]:
    output = subprocess.check_output(["ps", "-ax", "-o", "command="], text=True)
    return [line for line in output.splitlines() if "run_condition.py" in line and "adaptive_" in line]


def job_is_live(job: Job, commands: list[str]) -> bool:
    return any(str(job.logdir) in line for line in commands)


def log_text(job: Job) -> str:
    path = LOGS / f"{job.name}.attempt{job.attempts}.log"
    if job.attempts < 1 or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-6000:].lower()


def start(job: Job, python: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    job.attempts += 1
    handle = (LOGS / f"{job.name}.attempt{job.attempts}.log").open("w", encoding="utf-8")
    job.log_handle = handle
    job.proc = subprocess.Popen(
        job.command(python),
        cwd=ROOT,
        env=child_env(),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def finish(job: Job) -> str:
    assert job.proc is not None
    code = job.proc.wait()
    if job.log_handle is not None:
        job.log_handle.close()
        job.log_handle = None
    job.proc = None
    if job.complete():
        return "done"
    if "rate limit" in log_text(job) or "429" in log_text(job):
        job.not_before = time.time() + 30
        return "rate_limit"
    if code == 0:
        return "short"
    return f"exit {code}"


def snapshot(jobs: list[Job], running: list[Job], pending: list[Job], failed: list[dict]) -> None:
    payload = {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pending": len(pending),
        "running": [job.name for job in running],
        "finished_jobs": sum(1 for job in jobs if job.complete()),
        "failed": failed,
        "traces": {job.name: {"done": count_done(job.logdir), "expected": job.expected} for job in jobs},
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    done = sum(item["done"] for item in payload["traces"].values())
    expected = sum(item["expected"] for item in payload["traces"].values())
    print(
        f"{payload['updated']} traces {done}/{expected} "
        f"jobs_done {payload['finished_jobs']}/{len(jobs)} running {len(running)} pending {len(pending)}",
        flush=True,
    )


def run_jobs(jobs: list[Job], python: str) -> int:
    pending = [job for job in jobs if not job.complete()]
    print(f"jobs {len(jobs)} already_done {len(jobs) - len(pending)} to_run {len(pending)}", flush=True)
    running: list[Job] = []
    failed: list[dict] = []
    last_report = 0.0
    while pending or running:
        still_running = []
        for job in running:
            if job.proc is not None and job.proc.poll() is None:
                still_running.append(job)
                continue
            outcome = finish(job)
            if outcome != "done":
                print(
                    f"{job.name} {outcome} done {count_done(job.logdir)}/{job.expected} attempt {job.attempts}",
                    flush=True,
                )
                if job.attempts < MAX_ATTEMPTS:
                    pending.append(job)
                else:
                    failed.append(
                        {
                            "name": job.name,
                            "outcome": outcome,
                            "done": count_done(job.logdir),
                            "expected": job.expected,
                        }
                    )
        running = still_running

        commands = live_commands()
        adopted = [job for job in pending if job_is_live(job, commands)]
        api_running = sum(1 for job in running if job.kind == "api") + sum(1 for job in adopted if job.kind == "api")
        detector_running = sum(1 for job in running if job.kind == "detector") + sum(
            1 for job in adopted if job.kind == "detector"
        )
        available = free_percent()
        rest = []
        for job in pending:
            if job.complete():
                continue
            if time.time() < job.not_before:
                rest.append(job)
                continue
            blocked = job in adopted or available < MIN_FREE_PERCENT
            if not blocked and job.kind == "detector" and (
                available < MIN_FREE_PERCENT_DETECTOR or detector_running >= DETECTOR_SLOTS
            ):
                blocked = True
            if not blocked and job.kind == "api" and api_running >= API_SLOTS:
                blocked = True
            if blocked:
                rest.append(job)
                continue
            start(job, python)
            running.append(job)
            if job.kind == "api":
                api_running += 1
            else:
                detector_running += 1
            print(
                f"start {job.name} attempt {job.attempts} expected {job.expected} free_mem {available}%",
                flush=True,
            )
            last_report = 0.0
        pending = rest

        now = time.time()
        if now - last_report > 30:
            snapshot(jobs, running, pending, failed)
            last_report = now
        time.sleep(5)

    snapshot(jobs, running, pending, failed)
    if failed:
        print(json.dumps(failed, indent=2), flush=True)
        return 1
    print("CONFIRM_TRACES_DONE", flush=True)
    return 0


def assert_complete(selected: dict[str, dict]) -> None:
    from analysis.score import load_traces

    runs = CACHE if CACHE.is_dir() else STUDY
    traces = load_traces(runs, None)
    problems = []
    for defense, choice in selected.items():
        rows = [trace for trace in traces if trace["defense"] == defense and trace["attack"] == choice["variant"]]
        keys = [(trace["user_task"], trace["injection_task"], trace["repeat"]) for trace in rows]
        pairs = {(trace["user_task"], trace["injection_task"]) for trace in rows}
        if len(keys) != len(set(keys)):
            problems.append(f"{defense} has duplicate confirmation traces")
        if any(trace["repeat"] != 1 for trace in rows):
            problems.append(f"{defense} confirmation has a repeat other than 1")
        if len(pairs) != N_PAIRS or len(rows) != N_PAIRS:
            problems.append(f"{defense} has {len(rows)} confirmation traces ({len(pairs)} pairs), expected {N_PAIRS}")
    if problems:
        raise SystemExit("Confirmation is incomplete:\n" + "\n".join(problems))


def write_selection(selected: dict[str, dict], status: str, extra: dict | None = None) -> None:
    payload = {
        "model": MODEL_ID,
        "suite": SUITE,
        "benchmark_version": BENCHMARK,
        "repeats": 1,
        "n_pairs": N_PAIRS,
        "dev_log": str(DEV_LOG.relative_to(ROOT)),
        "selection_rule": (
            "Per defense, highest targeted ASR on the 10-pair dev slice. "
            "Ties break by mean targeted ASR of that variant across defenses, "
            "then by the lexicographically later variant name."
        ),
        "selected": selected,
        "status": status,
    }
    if extra:
        payload.update(extra)
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    SELECTION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def rescore() -> dict:
    from analysis.score import format_report, load_traces, score_traces, write_report

    runs = CACHE if CACHE.is_dir() else STUDY
    traces = load_traces(runs, None)
    report = score_traces(traces)
    report["runs"] = str(runs)
    report["confirmation"] = {
        "repeats": 1,
        "n_pairs": N_PAIRS,
        "recorded_on": time.strftime("%Y-%m-%d"),
        "selection": str(SELECTION.relative_to(ROOT)),
    }
    write_report(report, REPORT)
    print(format_report(report), flush=True)
    print(f"wrote {REPORT}", flush=True)
    return report


def verdict_summary(report: dict, selected: dict[str, dict]) -> dict:
    model = report["models"][MODEL_ID]
    summary = {}
    for defense, stats in model["defenses"].items():
        rules = stats["rules"]
        summary[defense] = {
            "variant": selected[defense]["variant"],
            "verdict": rules["verdict"],
            "adaptive_asr": stats["adaptive_asr"],
            "fixed_asr": stats["fixed_asr"],
            "adaptive_gap": stats["adaptive_gap"],
            "wilson_upper_95": stats["wilson_upper_95"].get(rules["adaptive_attack"]) if rules["adaptive_attack"] else None,
            "benign_utility": stats["benign_utility"],
            "n_pairs": stats["n_pairs"].get(rules["adaptive_attack"]) if rules["adaptive_attack"] else None,
        }
        if rules["verdict"] not in {"pass", "fail"}:
            raise SystemExit(f"{defense} verdict is {rules['verdict']}, expected pass or fail.")
        if summary[defense]["n_pairs"] != N_PAIRS:
            raise SystemExit(f"{defense} scored n_pairs={summary[defense]['n_pairs']}, expected {N_PAIRS}.")
    return summary


def self_check() -> None:
    try:
        select_variants(
            [
                {
                    "defense": "none",
                    "variant": "adaptive_paraphrase",
                    "targeted_asr": 0.2,
                    "n_pairs": 10,
                    "utility_under_attack": 0.6,
                }
            ]
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("missing defenses should abort")
    complete = []
    for defense in ("none", "spotlighting_with_delimiting", "transformers_pi_detector", "tool_filter"):
        paraphrase_asr = 0.0 if defense == "tool_filter" else 0.2
        complete.append(
            {
                "defense": defense,
                "variant": "adaptive_paraphrase",
                "targeted_asr": paraphrase_asr,
                "n_pairs": 10,
                "utility_under_attack": 0.5,
            }
        )
        for variant in ("adaptive_system_message", "adaptive_split", "adaptive_delimiter_spoof"):
            complete.append(
                {
                    "defense": defense,
                    "variant": variant,
                    "targeted_asr": 0.0,
                    "n_pairs": 10,
                    "utility_under_attack": 0.5,
                }
            )
    selected = select_variants(complete)
    for defense, choice in selected.items():
        if choice["variant"] != "adaptive_paraphrase":
            raise AssertionError((defense, choice))
    if len(user_tasks()) != N_USERS or N_PAIRS != 560:
        raise AssertionError("workspace pair count drifted")
    covered = [task for group in shard_tasks(user_tasks(), N_SHARDS) for task in group]
    if sorted(covered) != sorted(user_tasks()) or len(shard_tasks(user_tasks(), N_SHARDS)) != N_SHARDS:
        raise AssertionError("shards dropped a user task")
    live = select_variants(load_dev_rows())
    for defense, choice in live.items():
        if choice["variant"] != "adaptive_paraphrase":
            raise AssertionError((defense, choice))
        if defense == "tool_filter" and choice["dev_targeted_asr"] != 0.0:
            raise AssertionError(choice)
        if defense != "tool_filter" and choice["dev_targeted_asr"] != 0.2:
            raise AssertionError(choice)
    print("adaptive_confirm.py self-check passed")
    for defense, choice in live.items():
        print(f"{defense} {choice['variant']} dev_asr={choice['dev_targeted_asr']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true", help="Check variant selection, then exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected variants and shard commands, then exit.")
    parser.add_argument("--score-only", action="store_true", help="Rescore finished confirmation traces and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    selected = select_variants(load_dev_rows())
    if args.dry_run:
        python = python_executable()
        for defense, choice in selected.items():
            print(f"{defense} {choice['variant']} dev_asr={choice['dev_targeted_asr']} n={choice['dev_n_pairs']}")
        for job in make_jobs(selected):
            print(" ".join(job.command(python)))
        return 0
    if args.score_only:
        assert_complete(selected)
        report = rescore()
        summary = verdict_summary(report, selected)
        write_selection(selected, "complete", {"verdicts": summary})
        print(f"wrote {SELECTION}", flush=True)
        return 0

    write_selection(selected, "running")
    python = python_executable()
    code = run_jobs(make_jobs(selected), python)
    if code != 0:
        write_selection(selected, "incomplete")
        return code
    assert_complete(selected)
    report = rescore()
    summary = verdict_summary(report, selected)
    write_selection(selected, "complete", {"verdicts": summary})
    print(f"wrote {SELECTION}", flush=True)
    print("CONFIRM_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
