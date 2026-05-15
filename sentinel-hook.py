#!/usr/bin/env python3
"""sentinel-hook — Claude Code PreToolUse hook for Sentinel.

Zero third-party dependencies (stdlib only) so cold-start stays under 50ms.
Reads the PreToolUse JSON envelope on stdin, POSTs to the local Sentinel
daemon, translates the verdict into a Claude Code hook exit code, and prints
any reason text to stderr so Claude Code returns it to the model as the
tool-call error.

Failure mode: fails OPEN (exit 0). The agent is never blocked because the
daemon is down — that would be worse than no Sentinel at all.

Install:
    cp sentinel-hook.py ~/.local/bin/sentinel-hook
    chmod +x ~/.local/bin/sentinel-hook

Configure (~/.claude/settings.json):
    {
      "hooks": {
        "PreToolUse": [{
          "matcher": "*",
          "hooks": [{
            "type": "command",
            "command": "/Users/<you>/.local/bin/sentinel-hook"
          }]
        }]
      }
    }
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

DAEMON_URL = os.environ.get("SENTINEL_DAEMON_URL", "http://127.0.0.1:7777/detect")
TIMEOUT_SECS = float(os.environ.get("SENTINEL_TIMEOUT_SECS", "0.2"))  # 200ms fail-open budget
LOG_PATH = os.path.expanduser(os.environ.get("SENTINEL_HOOK_LOG", "~/.sentinel/hook.log"))


def log(msg: str) -> None:
    """Append a single line to the hook log; silent on disk errors."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except OSError:
        pass  # never let logging break the hook


def main() -> int:
    # 1. Read PreToolUse envelope from stdin
    try:
        raw = sys.stdin.read()
        envelope = json.loads(raw) if raw else {}
    except json.JSONDecodeError as e:
        log(f"stdin_parse_error: {e}")
        return 0  # fail-open

    tool_name = envelope.get("tool_name", "")
    tool_input = envelope.get("tool_input", {})
    session_id = envelope.get("session_id", envelope.get("conversation_id", "default"))

    if not tool_name:
        log("missing_tool_name")
        return 0  # nothing to check; fail-open

    payload = json.dumps(
        {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "session_id": session_id,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        DAEMON_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # 2. Call daemon with strict timeout
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        log(f"daemon_unreachable: {type(e).__name__}: {e}")
        return 0  # fail-open

    verdict = body.get("verdict", "ALLOW")
    confidence = body.get("confidence", 0.0)
    reason = body.get("reason", "")

    # 3. Translate verdict to exit code + stderr message
    #
    # Claude Code PreToolUse contract:
    #   exit 0           → allow tool to run unchanged
    #   exit 2 + stderr  → block; stderr text returned to model as tool error
    if verdict == "ALLOW":
        return 0

    if verdict in ("AUTO_CORRECT", "SUGGEST", "BLOCK"):
        # Emit reason via stderr so Claude reads it and retries with corrected plan
        print(reason, file=sys.stderr)
        log(f"intercepted: tool={tool_name} verdict={verdict} conf={confidence:.2f}")
        return 2

    # Unknown verdict — fail-open with a log line so we can debug
    log(f"unknown_verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
