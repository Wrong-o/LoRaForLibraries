import asyncio
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from nicegui import ui

from tasks import get_tasks, GLOBAL_CONTRACT


# --------- Config ---------
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful NiceGUI coding assistant. Generate a single-file NiceGUI app that satisfies the contract."
)
PORT_A = 8765
PORT_B = 8766


# --------- Utilities ---------
def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def mkdir_p(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def read_text(url: str, timeout: float = 5.0) -> Tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.getcode(), body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body
    except Exception:
        return 0, ""


def post_json(url: str, data: Dict[str, Any], timeout: float = 5.0) -> Tuple[int, str]:
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.getcode(), body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body
    except Exception:
        return 0, ""


def extract_code_from_text(text: str) -> str:
    start = None
    end = None
    fence = "```"
    lines = text.splitlines()
    code_lines: List[str] = []
    in_block = False
    for line in lines:
        if line.strip().startswith(fence):
            if not in_block:
                in_block = True
                # if language tag present, skip that line
                continue
            else:
                end = True
                break
        if in_block:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines).strip()
    return text.strip()


def enforce_run_args(code: str) -> str:
    """Ensure generated code runs with required NiceGUI run args and imports os.
    Strategy: comment out any existing ui.run(...) call(s) line-by-line (robust to multi-line),
    then append a single correct run call at the end. Also ensure `import os` exists.
    """
    # Ensure import os at top (avoid duplicates)
    if "import os" not in code:
        code = "import os\n" + code

    lines = code.splitlines()
    new_lines: List[str] = []
    commenting = False
    paren_depth = 0
    for line in lines:
        if not commenting and "ui.run(" in line:
            commenting = True
            # initialize depth count using parentheses on this line starting from the first ui.run(
            idx = line.find("ui.run(")
            # Count parentheses from idx
            segment = line[idx:]
            paren_depth = segment.count("(") - segment.count(")")
            new_lines.append("# " + line)
            if paren_depth <= 0:
                commenting = False
            continue
        if commenting:
            # Update depth across continued lines
            paren_depth += line.count("(") - line.count(")")
            new_lines.append("# " + line)
            if paren_depth <= 0:
                commenting = False
            continue
        new_lines.append(line)

    if commenting:
        # Close any dangling comment block
        commenting = False

    code_no_runs = "\n".join(new_lines).rstrip()
    # Append our enforced single-line run
    code_no_runs += "\n\nui.run(port=int(os.getenv('PORT', 8765)), show=False, reload=False)\n"
    return code_no_runs


def normalize_datatestid_props(code: str) -> str:
    """Rewrite common mistakes to proper NiceGUI props('data-testid=...').
    - Replace .data_testid('X') => .props('data-testid=X')
    - Replace .data_testid("X") => .props('data-testid=X')
    - Replace set_data_testid variants if any
    """
    code = re.sub(r"\.data_testid\(\s*'([^']+)'\s*\)", r".props('data-testid=\1')", code)
    code = re.sub(r'\.data_testid\(\s*"([^"]+)"\s*\)', r".props('data-testid=\1')", code)
    code = re.sub(r"\.set_data_testid\(\s*'([^']+)'\s*\)", r".props('data-testid=\1')", code)
    code = re.sub(r'\.set_data_testid\(\s*"([^"]+)"\s*\)', r".props('data-testid=\1')", code)
    return code


def build_prompt(task: Dict[str, Any]) -> str:
    items = [
        GLOBAL_CONTRACT,
        f"Task: {task['title']}",
        "Required data-testids: " + ", ".join(task.get("testids", [])),
        "Required interactions (in order):",
    ]
    for step in task.get("sequence", []):
        items.append("- " + json.dumps(step))
    items.append("Provide only runnable Python code in a single file with NiceGUI.")
    return "\n".join(items)


def ollama_chat_sync(model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": 0},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)


async def ollama_chat(model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    return await asyncio.to_thread(ollama_chat_sync, model, system_prompt, user_prompt)


def allowed_imports_okay(code: str) -> bool:
    # Simple heuristic: disallow mentions of common external libs
    banned = [
        "requests", "httpx", "fastapi", "flask", "django", "playwright", "selenium",
        "pandas", "numpy", "torch", "tensorflow",
    ]
    for lib in banned:
        if f"import {lib}" in code or f"from {lib} import" in code:
            return False
    # Must include nicegui
    return "nicegui" in code


async def generate_and_save_app(run_dir: str, model: str, task: Dict[str, Any], attempt: int,
                                system_prompt: str) -> Tuple[str, Dict[str, Any]]:
    prompt = build_prompt(task)
    result = await ollama_chat(model, system_prompt, prompt)
    raw_text = result.get("message", {}).get("content", "")
    code = extract_code_from_text(raw_text)
    code = enforce_run_args(code)
    code = normalize_datatestid_props(code)
    model_dir = os.path.join(run_dir, f"attempt_{attempt}")
    mkdir_p(model_dir)
    write_text(os.path.join(model_dir, "raw.txt"), raw_text)
    write_text(os.path.join(model_dir, "app.py"), code)
    return model_dir, result


async def start_app_subprocess(app_dir: str, port: int) -> Tuple[Any, Any, Any]:
    env = dict(os.environ)
    env["PORT"] = str(port)
    stdout_path = os.path.join(app_dir, "stdout.txt")
    stderr_path = os.path.join(app_dir, "stderr.txt")
    stdout_f = open(stdout_path, "w")
    stderr_f = open(stderr_path, "w")
    # Use sys.executable to ensure the same interpreter
    proc = await asyncio.create_subprocess_exec(
        sys.executable, os.path.join(app_dir, "app.py"),
        stdout=stdout_f, stderr=stderr_f, env=env, cwd=app_dir
    )
    return proc, stdout_f, stderr_f


async def wait_for_startup(port: int, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    url_health = f"http://127.0.0.1:{port}/health"
    url_root = f"http://127.0.0.1:{port}/"
    while time.time() < deadline:
        code1, _ = await asyncio.to_thread(read_text, url_health, 2.0)
        code2, _ = await asyncio.to_thread(read_text, url_root, 2.0)
        if code1 == 200 and code2 == 200:
            return True
        await asyncio.sleep(0.25)
    return False


def png_1x1_base64() -> str:
    # A 1x1 transparent PNG
    b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/aqQbE4AAAAASUVORK5CYII="
    )
    return b64


async def probe_actions(port: int, task: Dict[str, Any], timeout_interaction: int, timeout_background: int,
                        log_list: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    base = f"http://127.0.0.1:{port}"

    async def post(path: str, data: Dict[str, Any], t: float = 5.0) -> Tuple[int, str]:
        code, body = await asyncio.to_thread(post_json, base + path, data, t)
        log_list.append({"req": {"path": path, "data": data}, "resp": {"code": code, "body": body}})
        return code, body

    async def get_text(testid: str) -> str:
        code, body = await asyncio.to_thread(read_text, f"{base}/test/text?testid={urllib.parse.quote(testid)}", 5.0)
        try:
            obj = json.loads(body)
            return obj.get("text", "")
        except Exception:
            return ""

    # reset
    await post("/test/reset", {})

    for step in task.get("sequence", []):
        action = step.get("action")
        if action == "reset":
            await post("/test/reset", {})
        elif action == "click":
            testid = step["testid"]
            await post("/test/click", {"testid": testid})
        elif action == "input":
            await post("/test/input", {"testid": step["testid"], "value": step.get("value", "")})
        elif action == "upload_png_1x1":
            payload = {
                "testid": step["testid"],
                "filename": "tiny.png",
                "content_b64": png_1x1_base64(),
            }
            await post("/test/upload", payload, timeout_interaction)
        elif action == "assert_text_equals":
            text = await get_text(step["testid"])
            if text != step.get("value", ""):
                errors.append(f"assert_text_equals failed for {step['testid']}: got '{text}'")
        elif action == "assert_text_contains":
            text = await get_text(step["testid"])
            if step.get("value", "") not in text:
                errors.append(f"assert_text_contains failed for {step['testid']}")
        elif action == "assert_text_in":
            text = await get_text(step["testid"])
            if text not in set(step.get("values", [])):
                errors.append(f"assert_text_in failed for {step['testid']} -> '{text}'")
        elif action == "assert_nonempty_assistant":
            text = await get_text(step["testid"])
            if len(text.strip()) < 3:
                errors.append("assistant reply empty")
        elif action == "assert_list_contains":
            text = await get_text(step["testid"])
            if step.get("value", "") not in text:
                errors.append("list missing value")
        elif action == "assert_list_not_contains":
            text = await get_text(step["testid"])
            if step.get("value", "") in text:
                errors.append("list still contains value")
        elif action == "assert_progress_done":
            # Poll status until done or timeout_background
            deadline = time.time() + timeout_background
            ok = False
            while time.time() < deadline:
                status_text = await get_text(step.get("status_testid", "status-label"))
                if status_text.strip().lower() == "done":
                    ok = True
                    break
                await asyncio.sleep(0.25)
            if not ok:
                errors.append("progress did not complete")
        elif action == "assert_table_filtered_sorted":
            text = await get_text(step["testid"])
            filter_val = step.get("filter", "").lower()
            rows = [line.strip() for line in text.splitlines() if line.strip()]
            filtered = [r for r in rows if filter_val in r.lower()]
            if not filtered:
                errors.append("table filter produced no rows")
            # simplistic order check
            asc = sorted(filtered)
            if filtered and filtered[0] != asc[0]:
                errors.append("table not sorted ascending")
        else:
            errors.append(f"unknown action {action}")

    return len(errors) == 0, errors


def compute_score(started: bool, spec_ok: bool, interactions_ok: bool, code_ok: bool, quality_ok: bool) -> float:
    score = 0.0
    score += 1.5 if started else 0.0  # 30% of 5
    score += 1.5 if spec_ok else 0.0  # 30% of 5
    score += 1.0 if interactions_ok else 0.0  # 20% of 5
    score += 0.5 if code_ok else 0.0  # 10% of 5
    score += 0.5 if quality_ok else 0.0  # 10% of 5
    return score


@dataclass
class ModelRunResult:
    score: float
    started: bool
    spec_ok: bool
    interactions_ok: bool
    code_ok: bool
    quality_ok: bool
    errors: List[str] = field(default_factory=list)
    attempt: int = 1


async def evaluate_model(app_dir: str, port: int, task: Dict[str, Any], timeouts: Dict[str, int],
                         logs: List[Dict[str, Any]]) -> ModelRunResult:
    started = await wait_for_startup(port, timeouts.get("startup", 30))
    spec_ok = False
    interactions_ok = False
    code_ok = False
    quality_ok = True
    errors: List[str] = []

    if not started:
        return ModelRunResult(0.0, False, False, False, False, False, ["startup timeout"])  # type: ignore[arg-type]

    # spec: endpoints present & required testids (best-effort via /test/text not 400)
    base = f"http://127.0.0.1:{port}"
    code, body = await asyncio.to_thread(read_text, f"{base}/health", 5.0)
    spec_ok = code == 200 and "ok" in body

    action_logs: List[Dict[str, Any]] = []
    ok, errs = await probe_actions(port, task, timeouts.get("interaction", 5), timeouts.get("background", 15), action_logs)
    interactions_ok = ok
    errors.extend(errs)
    logs.extend(action_logs)

    # code constraints and quality proxy
    try:
        code_text = open(os.path.join(app_dir, "app.py"), "r", encoding="utf-8").read()
        code_ok = allowed_imports_okay(code_text)
    except Exception:
        code_ok = False

    score = compute_score(started, spec_ok, interactions_ok, code_ok, quality_ok)
    return ModelRunResult(score, started, spec_ok, interactions_ok, code_ok, quality_ok, errors)


# --------- UI Orchestrator ---------
tasks_list = get_tasks()


class SuiteRunner:
    def __init__(self) -> None:
        self.model_a: str = "nicegui-lora:latest"
        self.model_b: str = "qwen3-coder:latest"
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT
        self.ts: str = now_ts()
        self.current_task_index: int = -1
        self.next_event: Optional[asyncio.Event] = None
        self.results: List[Dict[str, Any]] = []
        self.iframes_container = None
        self.status_container = None
        self.results_container = None

    async def run_suite(self) -> None:
        for idx, task in enumerate(tasks_list):
            self.current_task_index = idx
            # alternate order: even A->B, odd B->A
            order: List[Tuple[str, int, str]]
            if idx % 2 == 0:
                order = [(self.model_a, PORT_A, "A"), (self.model_b, PORT_B, "B")]
            else:
                order = [(self.model_b, PORT_A, "B"), (self.model_a, PORT_B, "A")]

            base_dir = os.path.dirname(os.path.abspath(__file__))
            task_dir = os.path.join(base_dir, "runs", self.ts, task["id"])  # absolute under project
            mkdir_p(task_dir)

            # Generation
            if self.status_container is not None:
                self.status_container.clear()
                with self.status_container:
                    ui.label(f"Generating code for {task['title']}...")
            gen_dirs: Dict[str, str] = {}
            gen_meta: Dict[str, Dict[str, Any]] = {}
            for model_name, port, tag in order:
                model_dir = os.path.join(task_dir, tag)
                mkdir_p(model_dir)
                attempt = 1
                try:
                    subdir, meta = await generate_and_save_app(model_dir, model_name, task, attempt, self.system_prompt)
                    gen_dirs[tag] = subdir
                    gen_meta[tag] = meta
                except Exception as e:
                    write_text(os.path.join(model_dir, "error.txt"), f"generation error: {e}")
                    if self.status_container is not None:
                        with self.status_container:
                            ui.label(f"Generation failed for {tag}: {e}").classes('text-red-600')
                    gen_dirs[tag] = model_dir

            # Start both apps concurrently
            procs: Dict[str, Tuple[Any, Any, Any]] = {}
            for model_name, port, tag in order:
                app_dir = gen_dirs[tag]
                try:
                    proc, out_f, err_f = await start_app_subprocess(app_dir, port)
                    procs[tag] = (proc, out_f, err_f)
                except Exception as e:
                    write_text(os.path.join(app_dir, "start_error.txt"), f"start error: {e}")
                    if self.status_container is not None:
                        with self.status_container:
                            ui.label(f"Start failed for {tag}: {e}").classes('text-red-600')

            # Wait for startup
            started_a, started_b = await asyncio.gather(
                wait_for_startup(PORT_A, task["timeouts"]["startup"]),
                wait_for_startup(PORT_B, task["timeouts"]["startup"]),
            )
            if self.status_container is not None:
                self.status_container.clear()
                with self.status_container:
                    ui.label(f"Startup A: {'OK' if started_a else 'FAIL'}  B: {'OK' if started_b else 'FAIL'}")
                    # If failed, show stderr tail for quick diagnosis
                    for tag, started in ((order[0][2], started_a), (order[1][2], started_b)):
                        if not started:
                            try:
                                err_path = os.path.join(gen_dirs[tag], "stderr.txt")
                                tail = open(err_path, 'r', encoding='utf-8').read()[-500:]
                                ui.label(f"{tag} stderr tail:").classes('text-red-600')
                                ui.label(tail).classes('text-xs text-red-600 whitespace-pre-wrap')
                            except Exception:
                                pass
            if not (started_a or started_b):
                # Skip to next task if neither started
                continue

            # Show manual iframes
            if self.iframes_container is not None:
                self.iframes_container.clear()
                with self.iframes_container:
                    ui.label(f"Manual review: {task['title']}")
                    with ui.row().classes('w-full gap-4'):
                        ui.html(f'<iframe src="http://127.0.0.1:{PORT_A}/" style="width:100%;height:420px;border:1px solid #ddd;"></iframe>', sanitize=False).classes('flex-1')
                        ui.html(f'<iframe src="http://127.0.0.1:{PORT_B}/" style="width:100%;height:420px;border:1px solid #ddd;"></iframe>', sanitize=False).classes('flex-1')
                    next_btn = ui.button("Next test")
                    self.next_event = asyncio.Event()
                    next_btn.on("click", lambda e: self.next_event.set() if self.next_event else None)

            if self.next_event is not None:
                await self.next_event.wait()
                self.next_event = None

            # Automated probes
            logs_a: List[Dict[str, Any]] = []
            logs_b: List[Dict[str, Any]] = []

            res_a, res_b = await asyncio.gather(
                evaluate_model(gen_dirs[order[0][2]], PORT_A, task, task["timeouts"], logs_a),
                evaluate_model(gen_dirs[order[1][2]], PORT_B, task, task["timeouts"], logs_b),
            )

            # Optional re-generation if < 3.5
            async def maybe_regen(tag: str, res: ModelRunResult) -> ModelRunResult:
                if res.score >= 3.5:
                    return res
                model_name = order[0][0] if tag == order[0][2] else order[1][0]
                parent_dir = os.path.join(task_dir, tag)
                subdir, _ = generate_and_save_app(parent_dir, model_name, task, 2, self.system_prompt)
                # restart app on same port
                # Kill old proc is complex since we used asyncio subprocess; to simplify, we skip auto-regen live run
                # and just keep the first attempt's score for now
                return res

            res_a = await maybe_regen(order[0][2], res_a)
            res_b = await maybe_regen(order[1][2], res_b)

            # Archive logs
            write_text(os.path.join(gen_dirs[order[0][2]], "probe_logs.jsonl"), "\n".join(json.dumps(x) for x in logs_a))
            write_text(os.path.join(gen_dirs[order[1][2]], "probe_logs.jsonl"), "\n".join(json.dumps(x) for x in logs_b))

            # Collect results
            self.results.append({
                "task_id": task["id"],
                "task_title": task["title"],
                "A_score": res_a.score,
                "B_score": res_b.score,
            })

            # Kill subprocesses (best-effort)
            for tag, (proc, out_f, err_f) in procs.items():
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=3.0)
                    except Exception:
                        proc.kill()
                except Exception:
                    pass
                try:
                    out_f.close()
                except Exception:
                    pass
                try:
                    err_f.close()
                except Exception:
                    pass

            # Update status UI
            if self.status_container is not None:
                self.status_container.clear()
                with self.status_container:
                    ui.label(f"Completed: {task['title']} -> A {res_a.score:.2f}, B {res_b.score:.2f}")

        # Summary table
        if self.results_container is not None:
            self.results_container.clear()
            with self.results_container:
                with ui.row().classes('w-full font-semibold'):
                    ui.label('Task').classes('w-2/3')
                    ui.label('A score').classes('w-1/6')
                    ui.label('B score').classes('w-1/6')
                for r in self.results:
                    with ui.row().classes('w-full'):
                        ui.label(r["task_title"]).classes('w-2/3')
                        ui.label(f"{r['A_score']:.2f}").classes('w-1/6')
                        ui.label(f"{r['B_score']:.2f}").classes('w-1/6')


runner = SuiteRunner()


with ui.column().classes('w-full p-4 gap-3'):
    ui.label('NiceGUI Model Evaluation Harness').classes('text-2xl font-bold')
    with ui.row().classes('w-full items-end gap-3'):
        model_a_input = ui.input(label='Model A', value=runner.model_a).classes('w-64')
        model_b_input = ui.input(label='Model B', value=runner.model_b).classes('w-64')
        system_prompt_input = ui.textarea(label='System prompt', value=runner.system_prompt).classes('w-[600px]')
        async def start_clicked() -> None:
            runner.model_a = model_a_input.value.strip()
            runner.model_b = model_b_input.value.strip()
            runner.system_prompt = system_prompt_input.value
            if runner.status_container is not None:
                runner.status_container.clear()
                with runner.status_container:
                    ui.label('Starting suite...')
            asyncio.create_task(runner.run_suite())
        ui.button('Start', on_click=lambda: asyncio.create_task(start_clicked()))

    runner.iframes_container = ui.column().classes('w-full gap-2')
    runner.status_container = ui.column().classes('w-full gap-2')
    ui.separator()
    ui.label('Results').classes('text-xl font-semibold')
    runner.results_container = ui.column().classes('w-full gap-1')

ui.run(reload=False)


