import sys, json, re

d = json.load(sys.stdin)
p = d.get("tool_input", {}).get("file_path", "")
if p and re.search(r"(?:^|/)data/.+\.csv$", p):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "Protected experimental data — do not edit CSVs in data/",
        }
    }))
