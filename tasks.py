"""
Task specifications for the NiceGUI-only evaluation harness.

All timings are per the contract: 30s startup, 5s interaction, 15s background.
These specs are used to construct prompts and to run automated probes via the
test endpoints that the generated apps must implement.
"""

from typing import Dict, List, Any


STARTUP_TIMEOUT_S: int = 30
INTERACTION_TIMEOUT_S: int = 5
BACKGROUND_TIMEOUT_S: int = 15


def _basic_counter() -> Dict[str, Any]:
    return {
        "id": "basic_counter",
        "title": "Basic: Counter Button",
        "level": "basic",
        "testids": [
            "increment-btn",
            "counter-label",
        ],
        "prompt": (
            "Create a page with a button that increments a counter label.\n"
            "Required elements with exact data-testid:\n"
            "- increment-btn (button)\n"
            "- counter-label (label)\n"
            "Required interactions: clicking increment-btn increases counter-label numeric text by 1.\n"
        ),
        "sequence": [
            {"action": "reset"},
            {"action": "click", "testid": "increment-btn"},
            {"action": "click", "testid": "increment-btn"},
            {"action": "click", "testid": "increment-btn"},
            {"action": "assert_text_equals", "testid": "counter-label", "value": "3"},
        ],
        "timeouts": {
            "startup": STARTUP_TIMEOUT_S,
            "interaction": INTERACTION_TIMEOUT_S,
            "background": BACKGROUND_TIMEOUT_S,
        },
    }


def _basic_tabs() -> Dict[str, Any]:
    return {
        "id": "basic_tabs",
        "title": "Basic: Tabs Switch",
        "level": "basic",
        "testids": [
            "tab-a",
            "tab-b",
            "tab-panel",
        ],
        "prompt": (
            "Create a page with two tabs A and B using NiceGUI's ui.tabs with a bound value.\n"
            "Required elements with exact data-testid: tab-a, tab-b, tab-panel.\n"
            "Use a tabs component with value='A' initially and update selection using tabs.set_value('B') or by setting tabs.value = 'B'.\n"
            "Do NOT call methods on ui.tab items like set_selected (that API does not exist).\n"
            "Switching to B must update tab-panel to contain the text 'B'.\n"
        ),
        "sequence": [
            {"action": "reset"},
            {"action": "click", "testid": "tab-b"},
            {"action": "assert_text_contains", "testid": "tab-panel", "value": "B"},
        ],
        "timeouts": {
            "startup": STARTUP_TIMEOUT_S,
            "interaction": INTERACTION_TIMEOUT_S,
            "background": BACKGROUND_TIMEOUT_S,
        },
    }


def _basic_input_submit() -> Dict[str, Any]:
    return {
        "id": "basic_input_submit",
        "title": "Basic: Input + Validation + Submit",
        "level": "basic",
        "testids": [
            "name-input",
            "submit-btn",
            "error-label",
            "result-label",
        ],
        "prompt": (
            "Create a form with name input, validation, and submit.\n"
            "Required elements: name-input (input), submit-btn (button), error-label (label), result-label (label).\n"
            "Submitting empty should set error-label to 'Required'. Submitting 'Alice' should clear error and set result-label to 'Hello, Alice!'.\n"
        ),
        "sequence": [
            {"action": "reset"},
            {"action": "click", "testid": "submit-btn"},
            {"action": "assert_text_equals", "testid": "error-label", "value": "Required"},
            {"action": "input", "testid": "name-input", "value": "Alice"},
            {"action": "click", "testid": "submit-btn"},
            {"action": "assert_text_equals", "testid": "result-label", "value": "Hello, Alice!"},
            {"action": "assert_text_equals", "testid": "error-label", "value": ""},
        ],
        "timeouts": {
            "startup": STARTUP_TIMEOUT_S,
            "interaction": INTERACTION_TIMEOUT_S,
            "background": BACKGROUND_TIMEOUT_S,
        },
    }


def _intermediate_table() -> Dict[str, Any]:
    return {
        "id": "intermediate_table",
        "title": "Intermediate: Table with Sort/Filter",
        "level": "intermediate",
        "testids": [
            "filter-input",
            "sort-name",
            "table",
        ],
        "prompt": (
            "Create a table with a text filter and a sort by name.\n"
            "Required elements: filter-input (input), sort-name (button), table (table).\n"
            "Filtering with 'ap' must restrict rows to items containing 'ap' (case-insensitive). Sorting ascending puts alphabetically first row first.\n"
        ),
        "sequence": [
            {"action": "reset"},
            {"action": "input", "testid": "filter-input", "value": "ap"},
            {"action": "click", "testid": "sort-name"},
            {"action": "assert_table_filtered_sorted", "testid": "table", "filter": "ap", "column": "name", "order": "asc"},
        ],
        "timeouts": {
            "startup": STARTUP_TIMEOUT_S,
            "interaction": INTERACTION_TIMEOUT_S,
            "background": BACKGROUND_TIMEOUT_S,
        },
    }


def _intermediate_theme() -> Dict[str, Any]:
    return {
        "id": "intermediate_theme",
        "title": "Intermediate: Theme Toggle",
        "level": "intermediate",
        "testids": [
            "theme-toggle",
            "theme-label",
        ],
        "prompt": (
            "Create a theme toggle that flips between 'light' and 'dark'.\n"
            "Required elements: theme-toggle (switch or button), theme-label (label showing current).\n"
        ),
        "sequence": [
            {"action": "reset"},
            {"action": "click", "testid": "theme-toggle"},
            {"action": "assert_text_in", "testid": "theme-label", "values": ["light", "dark"]},
        ],
        "timeouts": {
            "startup": STARTUP_TIMEOUT_S,
            "interaction": INTERACTION_TIMEOUT_S,
            "background": BACKGROUND_TIMEOUT_S,
        },
    }


def _intermediate_upload() -> Dict[str, Any]:
    return {
        "id": "intermediate_upload",
        "title": "Intermediate: File Upload + Preview",
        "level": "intermediate",
        "testids": [
            "file-input",
            "image-preview",
        ],
        "prompt": (
            "Create an image upload input that previews the uploaded image.\n"
            "Required elements: file-input (upload), image-preview (image).\n"
            "The app must accept /test/upload with base64 PNG and set preview src; /test/text?testid=image-preview should return 'ready' once preview set.\n"
        ),
        "sequence": [
            {"action": "reset"},
            {"action": "upload_png_1x1", "testid": "file-input", "target": "image-preview"},
            {"action": "assert_text_equals", "testid": "image-preview", "value": "ready"},
        ],
        "timeouts": {
            "startup": STARTUP_TIMEOUT_S,
            "interaction": INTERACTION_TIMEOUT_S,
            "background": BACKGROUND_TIMEOUT_S,
        },
    }


def _advanced_chat() -> Dict[str, Any]:
    return {
        "id": "advanced_chat",
        "title": "Advanced: Chat-like Streaming UI",
        "level": "advanced",
        "testids": [
            "chat-input",
            "send-btn",
            "chat-log",
        ],
        "prompt": (
            "Create a simple chat UI that streams or appends assistant text after sending.\n"
            "Required elements: chat-input (input), send-btn (button), chat-log (container).\n"
            "Sending 'hi' must show the user message and a non-empty assistant reply within 15s.\n"
        ),
        "sequence": [
            {"action": "reset"},
            {"action": "input", "testid": "chat-input", "value": "hi"},
            {"action": "click", "testid": "send-btn"},
            {"action": "assert_text_contains", "testid": "chat-log", "value": "hi"},
            {"action": "assert_nonempty_assistant", "testid": "chat-log"},
        ],
        "timeouts": {
            "startup": STARTUP_TIMEOUT_S,
            "interaction": INTERACTION_TIMEOUT_S,
            "background": BACKGROUND_TIMEOUT_S,
        },
    }


def _advanced_progress() -> Dict[str, Any]:
    return {
        "id": "advanced_progress",
        "title": "Advanced: Progress Bar with Background Task",
        "level": "advanced",
        "testids": [
            "start-task-btn",
            "progress-bar",
            "status-label",
        ],
        "prompt": (
            "Create a progress bar that fills to 100% in a background task.\n"
            "Required elements: start-task-btn (button), progress-bar, status-label.\n"
            "Completion must set status-label to 'done' within 15s.\n"
        ),
        "sequence": [
            {"action": "reset"},
            {"action": "click", "testid": "start-task-btn"},
            {"action": "assert_progress_done", "testid": "progress-bar", "status_testid": "status-label"},
        ],
        "timeouts": {
            "startup": STARTUP_TIMEOUT_S,
            "interaction": INTERACTION_TIMEOUT_S,
            "background": BACKGROUND_TIMEOUT_S,
        },
    }


def _advanced_dialog_crud() -> Dict[str, Any]:
    return {
        "id": "advanced_dialog_crud",
        "title": "Advanced: Dialog CRUD",
        "level": "advanced",
        "testids": [
            "add-btn",
            "item-input",
            "save-btn",
            "items-list",
            "edit-btn-0",
            "delete-btn-0",
        ],
        "prompt": (
            "Create CRUD with a dialog: add, edit, delete items.\n"
            "Required elements with given data-testids.\n"
            "After add 'X', list shows X; edit first to 'Y', list shows Y; then delete first removes it.\n"
        ),
        "sequence": [
            {"action": "reset"},
            {"action": "click", "testid": "add-btn"},
            {"action": "input", "testid": "item-input", "value": "X"},
            {"action": "click", "testid": "save-btn"},
            {"action": "assert_list_contains", "testid": "items-list", "value": "X"},
            {"action": "click", "testid": "edit-btn-0"},
            {"action": "input", "testid": "item-input", "value": "Y"},
            {"action": "click", "testid": "save-btn"},
            {"action": "assert_list_contains", "testid": "items-list", "value": "Y"},
            {"action": "click", "testid": "delete-btn-0"},
            {"action": "assert_list_not_contains", "testid": "items-list", "value": "Y"},
        ],
        "timeouts": {
            "startup": STARTUP_TIMEOUT_S,
            "interaction": INTERACTION_TIMEOUT_S,
            "background": BACKGROUND_TIMEOUT_S,
        },
    }


def get_tasks() -> List[Dict[str, Any]]:
    return [
        _basic_counter(),
        _basic_tabs(),
        _basic_input_submit(),
        _intermediate_table(),
        _intermediate_theme(),
        _intermediate_upload(),
        _advanced_chat(),
        _advanced_progress(),
        _advanced_dialog_crud(),
    ]


GLOBAL_CONTRACT = """
Build a NiceGUI app that runs at port from env PORT (default 8765). 
Use only NiceGUI and Python standard library. Add data-testid attributes to key elements exactly as specified. 
The page at '/' must render immediately. Keep all state in-process; no files or databases. 
Provide the following HTTP JSON test endpoints using only NiceGUI routing: 
GET /health -> {"ok": true}; 
POST /test/reset; POST /test/click {testid}; POST /test/input {testid,value}; POST /test/upload {testid,filename,content_b64}; 
GET /test/text?testid=... returns current visible text for that element as {"text": "..."}. 
Return 400 for unknown testid. The code must be a single Python file runnable with 'python file.py'.
When running the app, do not auto-open a browser (set show=False) and disable auto-reload (set reload=False). Call ui.run(port=int(os.getenv('PORT', 8765)), show=False, reload=False).
Implementation constraints: only use official NiceGUI APIs; do not call non-existent methods on components. For tabs, use ui.tabs with a string value (e.g., value='A') and switch via tabs.set_value('B') or by updating tabs.value; do NOT call methods on ui.tab items like set_selected.
Set data-testid using NiceGUI's props API: element.props('data-testid=...'). Do NOT use element.data_testid(...).
"""


