"""Run the secondary conditions that sit outside the pass/fail grid.

Undefended ``ignore_previous`` is 40 workspace user tasks crossed with all 14
injection tasks, three repeats (1,680 traces). The local grid repeats the four
workspace defenses, benign and ``important_instructions``, three times, against
Llama 3.1 8B Instruct served on ``LOCAL_LLM_PORT``. Traces stay under the study
cache (``runs/study`` is a symlink there) so they sit next to the fixed-attack
grid. This script does not rewrite ``results/workspace_grid.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = Path.home() / "Library" / "Caches" / "cs120-runs" / "study"
LOGS = Path.home() / "Library" / "Caches" / "cs120-secondary-logs"
STATUS = Path.home() / "Library" / "Caches" / "cs120-secondary-status.json"
HF_HOME = str(Path.home() / "Library" / "Caches" / "cs120-hf")
MODELS = Path.home() / "Library" / "Caches" / "cs120-models"
IGNORE_REPORT = ROOT / "results" / "ignore_previous.json"
LOCAL_REPORT = ROOT / "results" / "local_grid.json"
MODEL = "GPT_4O_MINI_2024_07_18"
MODEL_ID = "gpt-4o-mini-2024-07-18"
LOCAL_MODEL = "LOCAL"
LOCAL_QUANT = "Q4_K_M"
LOCAL_MODEL_ID = "llama-3.1-8b-instruct-q4_k_m"
LOCAL_WEIGHTS = MODELS / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
BENCHMARK = "v1.2.2"
SUITE = "workspace"
N_USERS = 40
N_INJECTIONS = 14
N_PAIRS = N_USERS * N_INJECTIONS
REPEATS = 3
N_SHARDS = 4
API_SLOTS = 3
MAX_ATTEMPTS = 5
STALL_SECONDS = 8 * 60
MIN_FREE_PERCENT = 15
DEFENSES: tuple[str | None, ...] = (
    None,
    "spotlighting_with_delimiting",
    "transformers_pi_detector",
    "tool_filter",
)
DETECTOR = "transformers_pi_detector"
CONTEXT = 32768


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
    def __init__(
        self,
        *,
        name: str,
        defense: str | None,
        attack: str | None,
        tasks: list[str],
        logdir: Path,
        kind: str,
        model: str,
        model_id: str | None,
    ) -> None:
        self.name = name
        self.defense = defense
        self.attack = attack
        self.tasks = tasks
        self.logdir = logdir
        self.kind = kind
        self.model = model
        self.model_id = model_id
        self.attempts = 0
        self.stalls = 0
        self.proc: subprocess.Popen | None = None
        self.log_handle = None
        self.not_before = 0.0
        self.started = 0.0

    @property
    def expected(self) -> int:
        per_user = N_INJECTIONS if self.attack else 1
        return len(self.tasks) * per_user * REPEATS

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
            self.model,
            "--repeats",
            str(REPEATS),
            "--logdir",
            str(self.logdir),
            "--module-to-load",
            "defenses.http_timeout",
        ]
        if self.model_id:
            command.extend(["--model-id", self.model_id])
        if self.defense:
            command.extend(["--defense", self.defense])
        if self.attack:
            command.extend(["--attack", self.attack])
        for task in self.tasks:
            command.extend(["--user-task", task])
        return command


def make_ignore_jobs() -> list[Job]:
    jobs = []
    for index, tasks in enumerate(shard_tasks(user_tasks(), N_SHARDS)):
        logdir = CACHE / SUITE / "none" / "ignore_previous" / f"shard_{index}"
        jobs.append(
            Job(
                name=f"ignore-none-s{index}",
                defense=None,
                attack="ignore_previous",
                tasks=tasks,
                logdir=logdir,
                kind="api",
                model=MODEL,
                model_id=None,
            )
        )
    return jobs


def make_local_jobs(defenses: tuple[str | None, ...] = DEFENSES) -> list[Job]:
    jobs = []
    tasks = user_tasks()
    for defense in defenses:
        label = defense_label(defense)
        kind = "detector" if defense == DETECTOR else "local"
        for attack in (None, "important_instructions"):
            attack_label = attack or "benign"
            for index, group in enumerate(shard_tasks(tasks, N_SHARDS)):
                logdir = CACHE / SUITE / label / attack_label / f"local_shard_{index}"
                jobs.append(
                    Job(
                        name=f"local-{label}-{attack_label}-s{index}",
                        defense=defense,
                        attack=attack,
                        tasks=group,
                        logdir=logdir,
                        kind=kind,
                        model=LOCAL_MODEL,
                        model_id=LOCAL_MODEL_ID,
                    )
                )
    return jobs


def child_env() -> dict[str, str]:
    copied = os.environ.copy()
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            copied.setdefault(key.strip(), value.strip().strip("'\""))
    copied["HF_HOME"] = HF_HOME
    copied["PYTHONUNBUFFERED"] = "1"
    copied["TOKENIZERS_PARALLELISM"] = "false"
    return copied


def local_port() -> int:
    raw = child_env().get("LOCAL_LLM_PORT", "8000")
    try:
        return int(raw)
    except ValueError:
        return 8000


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
    return [line for line in output.splitlines() if "run_condition.py" in line]


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
    job.started = time.time()
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


def log_activity(job: Job) -> float:
    path = LOGS / f"{job.name}.attempt{job.attempts}.log"
    if path.is_file():
        return path.stat().st_mtime
    return job.started


def kill_group(job: Job) -> None:
    if job.proc is None:
        return
    try:
        os.killpg(job.proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        job.proc.kill()


def finish(job: Job) -> str:
    assert job.proc is not None
    code = job.proc.wait()
    if job.log_handle is not None:
        job.log_handle.close()
        job.log_handle = None
    job.proc = None
    if job.complete():
        return "done"
    text = log_text(job)
    if "rate limit" in text or "429" in text:
        job.not_before = time.time() + 30
        return "rate_limit"
    if code == 0:
        return "short"
    return f"exit {code}"


def snapshot(phase: str, jobs: list[Job], running: list[Job], pending: list[Job], failed: list[dict]) -> None:
    payload = {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": phase,
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
        f"{payload['updated']} {phase} traces {done}/{expected} "
        f"jobs_done {payload['finished_jobs']}/{len(jobs)} running {len(running)} pending {len(pending)}",
        flush=True,
    )


def slot_key(kind: str) -> str:
    """Local and detector jobs share one server, so they take the same slot."""
    if kind == "api":
        return "api"
    return "local"


def slot_limit(kind: str) -> int:
    if kind == "api":
        return API_SLOTS
    return 1


def run_jobs(phase: str, jobs: list[Job], python: str) -> int:
    pending = [job for job in jobs if not job.complete()]
    print(f"{phase} jobs {len(jobs)} already_done {len(jobs) - len(pending)} to_run {len(pending)}", flush=True)
    running: list[Job] = []
    failed: list[dict] = []
    last_report = 0.0
    while pending or running:
        still_running = []
        for job in running:
            if job.proc is not None and job.proc.poll() is None:
                idle = time.time() - max(job.started, log_activity(job))
                if idle <= STALL_SECONDS:
                    still_running.append(job)
                    continue
                print(f"{job.name} stalled {int(idle)}s with no log output; restarting the shard", flush=True)
                kill_group(job)
                try:
                    job.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(job.proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        job.proc.kill()
                    job.proc.wait(timeout=10)
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
        counts = {"api": 0, "local": 0}
        for job in running + adopted:
            key = slot_key(job.kind)
            counts[key] = counts.get(key, 0) + 1
        available = free_percent()
        rest = []
        for job in pending:
            if job.complete() or time.time() < job.not_before:
                if not job.complete():
                    rest.append(job)
                continue
            memory_floor = 45 if job.kind == "detector" else MIN_FREE_PERCENT
            blocked = (
                job in adopted
                or available < memory_floor
                or counts.get(slot_key(job.kind), 0) >= slot_limit(job.kind)
            )
            if blocked:
                rest.append(job)
                continue
            start(job, python)
            running.append(job)
            key = slot_key(job.kind)
            counts[key] = counts.get(key, 0) + 1
            print(
                f"start {job.name} attempt {job.attempts} expected {job.expected} free_mem {available}%",
                flush=True,
            )
            last_report = 0.0
        pending = rest

        now = time.time()
        if now - last_report > 30:
            snapshot(phase, jobs, running, pending, failed)
            last_report = now
        time.sleep(5)

    snapshot(phase, jobs, running, pending, failed)
    if failed:
        print(json.dumps(failed, indent=2), flush=True)
        return 1
    print(f"{phase.upper()}_TRACES_DONE", flush=True)
    return 0


def attack_rows(traces: list[dict], model: str, defense: str, attack: str) -> list[dict]:
    return [
        trace
        for trace in traces
        if trace["model"] == model and trace["defense"] == defense and trace["attack"] == attack
    ]


def pair_summary(rows: list[dict]) -> dict:
    from analysis.score import average_repeats, wilson_upper

    averaged = average_repeats(rows, ("utility", "injection_succeeded"))
    n_pairs = len(averaged)
    successes = sum(row["injection_succeeded"] for row in averaged)
    utility = sum(row["utility"] for row in averaged) / n_pairs if n_pairs else None
    asr = successes / n_pairs if n_pairs else None
    per_repeat = {}
    for repeat in range(1, REPEATS + 1):
        repeat_rows = [row for row in rows if row["repeat"] == repeat]
        count = len(repeat_rows)
        per_repeat[str(repeat)] = {
            "n": count,
            "utility": (sum(row["utility"] for row in repeat_rows) / count) if count else None,
            "targeted_asr": (sum(row["injection_succeeded"] for row in repeat_rows) / count) if count else None,
        }
    keys = [(row["user_task"], row["injection_task"], row["repeat"]) for row in rows]
    return {
        "n_traces": len(rows),
        "n_pairs": n_pairs,
        "duplicate_traces": len(keys) != len(set(keys)),
        "utility_under_attack": utility,
        "targeted_asr": asr,
        "wilson_upper_95": wilson_upper(successes, n_pairs) if n_pairs else None,
        "per_repeat": per_repeat,
    }


def fixed_attack_reference() -> dict | None:
    path = ROOT / "results" / "workspace_grid.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    stats = report.get("models", {}).get(MODEL_ID, {}).get("defenses", {}).get("none", {})
    asr = stats.get("targeted_asr", {}).get("important_instructions")
    utility = stats.get("utility_under_attack", {}).get("important_instructions")
    if asr is None:
        return None
    return {"attack": "important_instructions", "targeted_asr": asr, "utility_under_attack": utility}


def write_ignore_report() -> dict:
    from analysis.score import load_traces

    traces = load_traces(CACHE, None)
    rows = attack_rows(traces, MODEL_ID, "none", "ignore_previous")
    summary = pair_summary(rows)
    if summary["n_pairs"] != N_PAIRS or summary["n_traces"] != N_PAIRS * REPEATS or summary["duplicate_traces"]:
        raise SystemExit(
            f"ignore_previous is incomplete: {summary['n_traces']} traces, {summary['n_pairs']} pairs"
        )
    reference = fixed_attack_reference()
    gap = None
    if reference is not None and summary["targeted_asr"] is not None:
        gap = summary["targeted_asr"] - reference["targeted_asr"]
    payload = {
        "recorded_on": time.strftime("%Y-%m-%d"),
        "model": MODEL_ID,
        "suite": SUITE,
        "benchmark_version": BENCHMARK,
        "defense": None,
        "attack": "ignore_previous",
        "pairs": N_PAIRS,
        "repeats": REPEATS,
        "pair_sample": "all 560 workspace pairs; eval_pairs.json was not frozen",
        "scoring": "Repeats of a pair are averaged first. The Wilson interval uses that pair-level rate with n = 560.",
        "runs": str(CACHE),
        **summary,
        "important_instructions": reference,
        "asr_minus_important_instructions": gap,
    }
    IGNORE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    IGNORE_REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"ignore_previous ASR {summary['targeted_asr']:.1%} utility {summary['utility_under_attack']:.1%} "
        f"wilson {summary['wilson_upper_95']:.1%}",
        flush=True,
    )
    print(f"wrote {IGNORE_REPORT}", flush=True)
    return payload


def served_model_ids(port: int) -> list[str]:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item.get("id", "") for item in payload.get("data", [])]


def llama_server_bin() -> Path | None:
    for candidate in (Path("/opt/homebrew/bin/llama-server"), Path("/usr/local/bin/llama-server")):
        if candidate.is_file():
            return candidate
    return None


def ensure_local_server() -> dict:
    """Return server metadata. Start llama-server when the Q4_K_M weights are on disk and the port is free."""
    port = local_port()
    try:
        ids = served_model_ids(port)
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        ids = []
    if any(LOCAL_MODEL_ID in item or "Llama-3.1-8B-Instruct" in item for item in ids):
        return {"status": "running", "port": port, "model_id": LOCAL_MODEL_ID, "served_ids": ids, "quantization": LOCAL_QUANT}
    if ids:
        return {
            "status": "wrong_model",
            "port": port,
            "served_ids": ids,
            "expected_model_id": LOCAL_MODEL_ID,
        }
    binary = llama_server_bin()
    if binary is None or not LOCAL_WEIGHTS.is_file():
        return {
            "status": "unavailable",
            "port": port,
            "binary": str(binary) if binary else None,
            "weights": str(LOCAL_WEIGHTS),
            "weights_present": LOCAL_WEIGHTS.is_file(),
            "quantization": LOCAL_QUANT,
            "reason": (
                "No OpenAI-compatible server is listening, and the Q4_K_M Llama 3.1 8B Instruct "
                "weights are not on disk. Meta's weights are gated, so the local grid was not started."
            ),
        }
    LOGS.mkdir(parents=True, exist_ok=True)
    handle = (LOGS / "llama-server.log").open("w", encoding="utf-8")
    subprocess.Popen(
        [
            str(binary),
            "-m",
            str(LOCAL_WEIGHTS),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            str(CONTEXT),
            "-ngl",
            "99",
            "--alias",
            LOCAL_MODEL_ID,
            "-np",
            "1",
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            ids = served_model_ids(port)
        except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            time.sleep(2)
            continue
        if ids:
            return {
                "status": "started",
                "port": port,
                "model_id": LOCAL_MODEL_ID,
                "served_ids": ids,
                "quantization": LOCAL_QUANT,
                "weights": str(LOCAL_WEIGHTS),
                "context": CONTEXT,
            }
        time.sleep(2)
    return {"status": "failed_to_start", "port": port, "weights": str(LOCAL_WEIGHTS), "log": str(LOGS / "llama-server.log")}


def write_local_report(server: dict, status: str) -> dict:
    from analysis.score import load_traces

    model_key = f"local:{LOCAL_MODEL_ID}"
    traces = load_traces(CACHE, None) if CACHE.is_dir() else []
    defenses = {}
    complete = status == "complete"
    for defense in DEFENSES:
        label = defense_label(defense)
        benign = [trace for trace in traces if trace["model"] == model_key and trace["defense"] == label and trace["attack"] == "benign"]
        attacked = attack_rows(traces, model_key, label, "important_instructions")
        attack_summary = pair_summary(attacked) if attacked else None
        n_benign = len({trace["user_task"] for trace in benign})
        defenses[label] = {
            "benign_traces": len(benign),
            "n_user_tasks": n_benign,
            "benign_utility": (sum(trace["utility"] for trace in benign) / len(benign)) if benign else None,
            "important_instructions": attack_summary,
        }
        if attack_summary is None or attack_summary["n_pairs"] != N_PAIRS or attack_summary["n_traces"] != N_PAIRS * REPEATS:
            complete = False
        if n_benign != N_USERS or len(benign) != N_USERS * REPEATS:
            complete = False
    if status == "complete" and not complete:
        status = "incomplete"
    payload = {
        "recorded_on": time.strftime("%Y-%m-%d"),
        "model": LOCAL_MODEL_ID,
        "base_model": "Llama 3.1 8B Instruct",
        "quantization": LOCAL_QUANT,
        "suite": SUITE,
        "benchmark_version": BENCHMARK,
        "attack": "important_instructions",
        "defenses": [defense_label(item) for item in DEFENSES],
        "pairs": N_PAIRS,
        "repeats": REPEATS,
        "status": status,
        "server": server,
        "runs": str(CACHE),
        "results": defenses,
    }
    LOCAL_REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"local grid status {status}", flush=True)
    print(f"wrote {LOCAL_REPORT}", flush=True)
    return payload


def self_check() -> None:
    ignore = make_ignore_jobs()
    covered = [task for job in ignore for task in job.tasks]
    if sorted(covered, key=lambda task: int(task.rsplit("_", 1)[1])) != user_tasks() or len(ignore) != N_SHARDS:
        raise AssertionError("ignore_previous shards dropped a user task")
    if sum(job.expected for job in ignore) != N_PAIRS * REPEATS:
        raise AssertionError(sum(job.expected for job in ignore))
    command = " ".join(ignore[0].command(python_executable()))
    if "ignore_previous" not in command or "--defense" in command or "--model LOCAL" in command:
        raise AssertionError(command)
    local = make_local_jobs()
    if len(local) != len(DEFENSES) * 2 * N_SHARDS:
        raise AssertionError(len(local))
    if sum(job.expected for job in local if job.attack) != N_PAIRS * REPEATS * len(DEFENSES):
        raise AssertionError("local attacked count drifted")
    if sum(job.expected for job in local if job.attack is None) != N_USERS * REPEATS * len(DEFENSES):
        raise AssertionError("local benign count drifted")
    tool_filter = next(job for job in local if job.defense == "tool_filter")
    if "--model LOCAL" not in " ".join(tool_filter.command("python")) or LOCAL_MODEL_ID not in " ".join(tool_filter.command("python")):
        raise AssertionError(tool_filter.command("python"))
    print("secondary_runs.py self-check passed")
    print(f"ignore_previous {sum(job.expected for job in ignore)} traces in {len(ignore)} shards")
    print(f"local grid {sum(job.expected for job in local)} traces in {len(local)} shards")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true", help="Check shard coverage and commands, then exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the shard commands and exit.")
    parser.add_argument("--score-only", action="store_true", help="Write the reports from traces already on disk.")
    parser.add_argument(
        "--phase",
        choices=("ignore", "local", "all"),
        default="all",
        help="ignore runs only the undefended attack. local runs the Llama grid. all runs ignore, then local.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    python = python_executable()
    if args.dry_run:
        jobs = []
        if args.phase in {"ignore", "all"}:
            jobs.extend(make_ignore_jobs())
        if args.phase in {"local", "all"}:
            jobs.extend(make_local_jobs())
        for job in jobs:
            print(" ".join(job.command(python)))
        return 0
    if args.score_only:
        if args.phase in {"ignore", "all"}:
            write_ignore_report()
        if args.phase in {"local", "all"}:
            try:
                server = {"status": "running", "port": local_port(), "served_ids": served_model_ids(local_port())}
            except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
                server = {"status": "unavailable", "port": local_port()}
            write_local_report(server, "complete")
        return 0

    if args.phase in {"ignore", "all"}:
        code = run_jobs("ignore_previous", make_ignore_jobs(), python)
        if code != 0:
            return code
        write_ignore_report()
        print("IGNORE_PREVIOUS_DONE", flush=True)
    if args.phase in {"local", "all"}:
        server = ensure_local_server()
        if server["status"] not in {"running", "started"}:
            write_local_report(server, "not_started")
            print("LOCAL_GRID_NOT_STARTED", flush=True)
            return 0
        defenses = DEFENSES
        if free_percent() < 45:
            defenses = tuple(item for item in DEFENSES if item != DETECTOR)
            server["detector_skipped"] = (
                "transformers_pi_detector was not started. The Q4_K_M server leaves under 45% "
                "memory free, and that defense loads a second model of about 7 GB."
            )
            print(server["detector_skipped"], flush=True)
        code = run_jobs("local", make_local_jobs(defenses), python)
        write_local_report(server, "complete" if code == 0 else "incomplete")
        if code != 0:
            return code
        print("LOCAL_GRID_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
