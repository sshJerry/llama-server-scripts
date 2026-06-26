#!/usr/bin/env python3
"""Ornit + AgentWorld tandem  ^`^t zero external dependencies (stdlib only)."""

import json, os, sys, urllib.request

AGENT = "http://localhost:8080/v1/chat/completions"
WORLD = "http://localhost:8081/v1/chat/completions"

# Match the -c values in your run scripts
CTX_LIMIT = {AGENT: 131072, WORLD: 32768}

PROMPTS_DIR = os.path.expanduser("~/prompts")
DOMAINS = ["android", "mcp", "os", "search", "swe", "terminal", "web"]

DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "terminal"
if DOMAIN not in DOMAINS:
    print(f"Unknown domain '{DOMAIN}'. Available: {', '.join(DOMAINS)}")
    sys.exit(1)


def load_prompt(domain, filename="system_prompt.txt"):
    path = os.path.join(PROMPTS_DIR, domain, filename)
    with open(path) as f:
        return f.read()


WORLD_PROMPT = load_prompt(DOMAIN)
AGENT_SYSTEM = (
    f"You are an AI agent operating in a {DOMAIN} environment. "
    "Output ONLY the next action. No explanation, no markdown. "
    "If the task is complete, output 'DONE'."
)
MAX_TURNS = 20


def fmt_tok(n):
    if n >= 1000:
        return f"{n/1000:.1f}K"
    return str(n)


def chat(base_url, messages, max_tokens=2048):
    data = json.dumps({
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(base_url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
        content = body["choices"][0]["message"]["content"].strip()
        usage = body.get("usage", {})
        return content, usage


def maybe_refresh(messages, usage, limit):
    """If context is near full, trim history and return True if refreshed."""
    prompt_tok = usage.get("prompt_tokens", 0)
    pct = prompt_tok / limit if limit > 0 else 0

    if pct < 0.80:
        return False

    print(f"\n[!] Context {pct*100:.0f}% full ({fmt_tok(prompt_tok)}/{fmt_tok(limit)})  ^`^t trimming history")
    # Keep system prompt + last 4 messages (2 turns of action+observation)
    keep = messages[:1] + messages[-4:] if len(messages) > 5 else messages
    messages.clear()
    messages.extend(keep)
    return True


print(f"Domain: {DOMAIN}  |  Ornith ctx: {fmt_tok(CTX_LIMIT[AGENT])}  |  AgentWorld ctx: {fmt_tok(CTX_LIMIT[WORLD])}\n")

messages = [{"role": "system", "content": AGENT_SYSTEM}]

task = input("Task: ")
messages.append({"role": "user", "content": task})

for turn in range(1, MAX_TURNS + 1):
    action, usage_a = chat(AGENT, messages)

    refreshed = maybe_refresh(messages, usage_a, CTX_LIMIT[AGENT])

    if action.upper() == "DONE":
        print(f"\nTurn {turn}: Agent finished.")
        break

    p_tok = usage_a.get("prompt_tokens", 0)
    c_tok = usage_a.get("completion_tokens", 0)
    t_tok = usage_a.get("total_tokens", 0)
    tag = "[R]" if refreshed else ""
    print(f"[{turn}]{tag} Action  [{fmt_tok(p_tok)}/{fmt_tok(CTX_LIMIT[AGENT])} ctx, +{fmt_tok(c_tok)} gen]")

    # Print action (with truncation for very long actions)
    if len(action) > 2000:
        print(action[:2000] + "...(truncated)")
    else:
        print(action)

    observation, usage_w = chat(WORLD, [
        {"role": "system", "content": WORLD_PROMPT},
        {"role": "user", "content": action},
    ], max_tokens=32768)

    w_p = usage_w.get("prompt_tokens", 0)
    w_c = usage_w.get("completion_tokens", 0)
    print(f"[{turn}]  Env    [{fmt_tok(w_p)}/{fmt_tok(CTX_LIMIT[WORLD])} ctx, +{fmt_tok(w_c)} gen]")
    if len(observation) > 2000:
        print(observation[:2000] + "...(truncated)")
    else:
        print(observation)

    messages.append({"role": "assistant", "content": action})
    messages.append({"role": "user", "content": f"Environment output:\n{observation}\n\nWhat is your next action?"})

print("\nDone.")