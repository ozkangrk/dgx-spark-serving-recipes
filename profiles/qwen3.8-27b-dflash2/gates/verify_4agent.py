#!/usr/bin/env python3
import concurrent.futures
import json
import time
import urllib.request

URL = "http://127.0.0.1:8888/v1/chat/completions"
MODEL = "qwen38-27b-radixark-dflash2"


def post(payload, timeout=180):
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.load(response)
    body["_latency_s"] = time.monotonic() - started
    return body


def content_of(body):
    msg = body["choices"][0]["message"]
    return (msg.get("content") or "") + (msg.get("reasoning_content") or "") + (msg.get("reasoning") or "")


def arithmetic_smoke():
    body = post({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Return only the integer result of 37*41."}],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    text = content_of(body).strip()
    assert text == "1517", repr(text)
    return {"ok": True, "latency_s": body["_latency_s"], "answer": text, "usage": body.get("usage")}


def tool_smoke():
    body = post({
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is the weather in Istanbul? Use the provided tool."}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get city weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    msg = body["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    assert calls and calls[0]["function"]["name"] == "get_weather", json.dumps(msg)
    args = calls[0]["function"].get("arguments", "")
    assert "Istanbul" in args or "istanbul" in args, args
    return {"ok": True, "latency_s": body["_latency_s"], "tool_calls": calls, "usage": body.get("usage")}


def one_parallel(i):
    body = post({
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Write exactly this marker and nothing else: AGENT_{i}_OK"}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    text = content_of(body).strip()
    expected = f"AGENT_{i}_OK"
    assert text == expected, (expected, text)
    return {"agent": i, "ok": True, "latency_s": body["_latency_s"], "answer": text, "usage": body.get("usage")}


def main():
    result = {"arithmetic": arithmetic_smoke(), "tool": tool_smoke()}
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result["parallel"] = list(pool.map(one_parallel, range(1, 5)))
    result["parallel_wall_s"] = time.monotonic() - started
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
