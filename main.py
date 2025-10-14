from nicegui import ui
import asyncio
import httpx
import json
import time

# Ollama server settings
OLLAMA_URL = "http://localhost:11434/api/chat"

# --- Tooling (reused) --- #
from datetime import datetime

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time and date",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

FUNCTION_REGISTRY = {
    "get_current_time": get_current_time,
    "calculate": calculate,
}

# --- UI State --- #
model_a_input = None
model_b_input = None
system_prompt_input = None
temp_input = None
topp_input = None
tools_toggle = None
prompt_input = None

# Per-pane containers
left_output = None
right_output = None
left_metrics = None
right_metrics = None

# Utility: append text to a label-like container by resetting its content
class StreamBuffer:
    def __init__(self):
        self.text = ""
        self.token_count = 0
        self.first_token_time = None
        self.start_time = None

    def reset(self):
        self.text = ""
        self.token_count = 0
        self.first_token_time = None
        self.start_time = None

    def add_text(self, new_text: str):
        if not new_text:
            return
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.text += new_text
        # heuristic token count by splitting on whitespace
        self.token_count += len(new_text.split())

# Streaming logic per model
async def stream_model(model_name: str, system_prompt: str, user_prompt: str, use_tools: bool, temperature: float, top_p: float, output_container, metrics_container):
    # Build messages: enforce same system for fairness
    messages = [
        {"role": "system", "content": system_prompt.strip() if system_prompt.strip() else "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": model_name,
        "stream": True,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        }
    }
    if use_tools:
        payload["tools"] = TOOLS

    buffer = StreamBuffer()
    buffer.start_time = time.perf_counter()

    # Ground-truth metrics from Ollama (filled from final chunk when available)
    eval_count = None
    eval_duration_ns = None

    output_container.clear()
    metrics_container.clear()
    with output_container:
        status_label = ui.label("⏳ Waiting for response...").classes('bg-yellow-100 p-2 rounded')
        text_label = ui.label("").classes('whitespace-pre-wrap')

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as resp:
                status_label.set_text("🔄 Streaming...")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue

                    # Capture final metrics before breaking
                    if chunk.get("done"):
                        # These fields are present on the final chunk
                        eval_count = chunk.get("eval_count", eval_count)
                        eval_duration_ns = chunk.get("eval_duration", eval_duration_ns)
                        break

                    # Handle tool calls (local only, no follow-up roundtrip)
                    if use_tools and "tool_calls" in chunk.get("message", {}):
                        tool_calls = chunk["message"]["tool_calls"]
                        for call in tool_calls:
                            fname = call.get("function", {}).get("name")
                            args_raw = call.get("function", {}).get("arguments", "{}")
                            try:
                                args = args_raw if isinstance(args_raw, dict) else json.loads(args_raw)
                            except Exception:
                                args = {}
                            result = None
                            try:
                                if fname in FUNCTION_REGISTRY:
                                    result = FUNCTION_REGISTRY[fname](**args) if args else FUNCTION_REGISTRY[fname]()
                                else:
                                    result = f"Function {fname} not implemented"
                            except Exception as e:
                                result = f"Error executing function: {e}"
                            buffer.add_text(f"\n[Tool {fname}]: {result}\n")
                            text_label.set_text(buffer.text)
                        continue

                    # Normal content
                    part = chunk.get("message", {}).get("content", "")
                    if part:
                        buffer.add_text(part)
                        text_label.set_text(buffer.text)
                        await asyncio.sleep(0.01)

        # Fallback: if nothing was streamed, try a non-stream request to populate content
        if buffer.token_count == 0:
            try:
                non_stream_payload = dict(payload)
                non_stream_payload["stream"] = False
                async with httpx.AsyncClient(timeout=None) as client:
                    r = await client.post(OLLAMA_URL, json=non_stream_payload)
                    data = r.json()
                    content = data.get("message", {}).get("content", "")
                    if content:
                        buffer.add_text(content)
                        text_label.set_text(buffer.text)
                    # Try to read metrics from non-stream response as well
                    eval_count = data.get("eval_count", eval_count)
                    eval_duration_ns = data.get("eval_duration", eval_duration_ns)
            except Exception:
                pass

        total_time = time.perf_counter() - buffer.start_time
        ttft = (buffer.first_token_time - buffer.start_time) if buffer.first_token_time else None

        # Prefer Ollama's ground-truth metrics when available
        if eval_count is not None and eval_duration_ns is not None and eval_duration_ns > 0:
            eval_duration_s = eval_duration_ns / 1e9
            tokens = int(eval_count)
            tps = tokens / eval_duration_s if eval_duration_s > 0 else 0.0
            tokens_label = f"Tokens (model): {tokens}"
            tps_label = f"Throughput (model): {tps:.2f} tok/s"
        else:
            # Safe fallback to approximation
            tps = (buffer.token_count / total_time) if total_time > 0 else 0.0
            tokens_label = f"Tokens (approx): {buffer.token_count}"
            tps_label = f"Throughput (approx): {tps:.2f} tok/s"

        status_label.set_text("✅ Completed")
        metrics_container.clear()
        with metrics_container:
            ui.label(f"TTFT: {ttft:.2f}s" if ttft is not None else "TTFT: n/a").classes('text-gray-600')
            ui.label(f"Total time: {total_time:.2f}s").classes('text-gray-600')
            ui.label(tokens_label).classes('text-gray-600')
            ui.label(tps_label).classes('text-gray-600')
    except Exception as e:
        status_label.set_text("❌ Error")
        output_container.clear()
        with output_container:
            ui.label(f"Error: {e}").classes('bg-red-100 text-red-700 p-2 rounded')

# Compare handler
async def on_compare_clicked():
    model_a = model_a_input.value.strip()
    model_b = model_b_input.value.strip()
    system_prompt = system_prompt_input.value
    user_prompt = prompt_input.value.strip()
    # Only enable tools when toggle explicitly equals 'on'
    use_tools = (tools_toggle.value == 'on')
    try:
        temperature = float(temp_input.value)
    except Exception:
        temperature = 0.7
    try:
        top_p = float(topp_input.value)
    except Exception:
        top_p = 0.9

    if not user_prompt:
        return

    # Reset outputs
    left_output.clear()
    right_output.clear()
    left_metrics.clear()
    right_metrics.clear()

    # Run both streams concurrently
    await asyncio.gather(
        stream_model(model_a, system_prompt, user_prompt, use_tools, temperature, top_p, left_output, left_metrics),
        stream_model(model_b, system_prompt, user_prompt, use_tools, temperature, top_p, right_output, right_metrics),
    )

# Copy helpers
def copy_left():
    # Use JS clipboard to avoid extra deps
    ui.run_javascript(f"navigator.clipboard.writeText({json.dumps(left_output.default_slot.children[1].text)})")

def copy_right():
    ui.run_javascript(f"navigator.clipboard.writeText({json.dumps(right_output.default_slot.children[1].text)})")

# --- UI Layout --- #
with ui.column().classes('w-full p-4 gap-3'):
    ui.label('NiceGUI - Side-by-Side Model Comparison').classes('text-2xl font-bold')

    with ui.row().classes('w-full items-end gap-3'):
        model_a_input = ui.input(label='Model A', value='nicegui-lora:latest').classes('w-64')
        model_b_input = ui.input(label='Model B', value='qwen3-coder:latest').classes('w-64')
        temp_input = ui.input(label='temperature', value='0.7').classes('w-28')
        topp_input = ui.input(label='top_p', value='0.9').classes('w-28')
        tools_toggle = ui.toggle(['off', 'on'], value='off', on_change=None).props('dense').classes('')
        ui.label('tools').classes('text-gray-600')

    system_prompt_input = ui.textarea(label='System prompt (applied to both)', value='You are a helpful assistant. Focus on NiceGUI best practices and runnable examples.').classes('w-full')

    with ui.row().classes('w-full items-end gap-3'):
        prompt_input = ui.textarea(label='Prompt', placeholder='Describe a NiceGUI task to compare...').classes('w-full')
        ui.button('Compare', on_click=lambda: asyncio.create_task(on_compare_clicked())).classes('')
        ui.button('Clear', on_click=lambda: (left_output.clear(), right_output.clear(), left_metrics.clear(), right_metrics.clear(), prompt_input.set_value('')))

    with ui.row().classes('w-full gap-4 flex flex-row'):
        with ui.column().classes('flex-1 gap-2 min-w-0'):
            ui.label('Model A Output').classes('font-semibold')
            left_output = ui.column().classes('min-h-[200px] p-2 bg-gray-50 rounded')
            left_metrics = ui.column().classes('gap-1')
            ui.button('Copy A', on_click=copy_left).props('flat size=sm')
        with ui.column().classes('flex-1 gap-2 min-w-0'):
            ui.label('Model B Output').classes('font-semibold')
            right_output = ui.column().classes('min-h-[200px] p-2 bg-gray-50 rounded')
            right_metrics = ui.column().classes('gap-1')
            ui.button('Copy B', on_click=copy_right).props('flat size=sm')

ui.run()

