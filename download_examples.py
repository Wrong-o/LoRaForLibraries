import requests, json, os

REPO_API = "https://api.github.com/repos/zauberzeug/nicegui/contents/examples"
OUTPUT_FILE = "nicegui_lora_dataset.json"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # set this in your shell

HEADERS = {}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

dataset = []

def fetch_files(url, path=""):
    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200:
        print(f"Failed to fetch {url}: {res.status_code}")
        return
    
    try:
        items = res.json()
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from {url}")
        return
    
    if not isinstance(items, list):
        print(f"Unexpected JSON structure at {url}")
        return

    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "file" and item.get("name", "").endswith(".py"):
            code_url = item.get("download_url")
            if not code_url:
                continue
            code_res = requests.get(code_url, headers=HEADERS)
            if code_res.status_code != 200:
                continue
            code = code_res.text.strip()
            instruction = f"Example from NiceGUI: {path + item['name'].replace('.py','')}"
            dataset.append({"instruction": instruction, "response": code})
        elif item.get("type") == "dir":
            fetch_files(item.get("url"), path + item["name"] + "/")

# Start recursion
fetch_files(REPO_API)

# Save JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Saved {len(dataset)} examples to {OUTPUT_FILE}")
