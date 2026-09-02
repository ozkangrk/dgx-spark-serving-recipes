#!/usr/bin/env python3
"""C1/C2/C3 streaming benchmark with usage-only accounting."""
import argparse, concurrent.futures, json, statistics, time, urllib.request, uuid
from pathlib import Path

PROMPTS = [
    "Implement a complete Python LRUCache class using OrderedDict, including get, put, delete, len, iteration, type hints, docstrings, and a unittest suite. Output only one Python code block. Continue adding meaningful edge-case tests until the token limit.",
    "Implement a complete Python rate limiter with token-bucket semantics, monotonic time injection, thread safety, type hints, docstrings, and a unittest suite. Output only one Python code block. Continue adding meaningful edge-case tests until the token limit.",
    "Implement a complete Python DAG utility with cycle detection, topological sort, dependency validation, type hints, docstrings, and a unittest suite. Output only one Python code block. Continue adding meaningful edge-case tests until the token limit.",
]

def one(endpoint, model, level, repeat, slot):
    nonce = uuid.uuid4().hex
    prompt = f"UNIQUE_NONCE_{nonce}\n" + PROMPTS[slot % len(PROMPTS)]
    payload = {
        "model": model,
        "messages": [{"role":"user","content":prompt}],
        "temperature": 0,
        "max_tokens": 512,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(endpoint + "/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json","Authorization":"Bearer local-key"})
    t0=time.perf_counter(); first=None; usage=None; finish=None; chars=0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line=raw.decode("utf-8").rstrip("\r\n")
            if not line.startswith("data:"): continue
            data=line[5:].lstrip()
            if data=="[DONE]": break
            obj=json.loads(data)
            if obj.get("usage") is not None: usage=obj["usage"]
            choices=obj.get("choices")
            if choices:
                finish=choices[0].get("finish_reason") or finish
                delta=choices[0].get("delta") or {}
                text=delta.get("content")
                if isinstance(text,str) and text:
                    now=time.perf_counter(); first=first or now; chars += len(text)
    t1=time.perf_counter()
    if not isinstance(usage,dict): raise RuntimeError("missing usage")
    pt=usage.get("prompt_tokens"); ct=usage.get("completion_tokens"); total=usage.get("total_tokens")
    if type(pt) is not int or type(ct) is not int or type(total) is not int or total != pt+ct: raise RuntimeError("malformed usage")
    if first is None: raise RuntimeError("no content delta")
    decode_window=t1-first
    return {
        "level":level,"repeat":repeat,"slot":slot,"nonce":nonce,
        "prompt_tokens":pt,"completion_tokens":ct,"finish_reason":finish,"chars":chars,
        "ttft_s":round(first-t0,6),"e2e_s":round(t1-t0,6),
        "decode_tokens_per_s":round((ct-1)/decode_window,4) if ct>1 and decode_window>0 else "UNMEASURABLE",
        "e2e_output_tokens_per_s":round(ct/(t1-t0),4),"start":t0,"end":t1,
        "pass": ct>=400,
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--endpoint",default="http://127.0.0.1:8888/v1"); ap.add_argument("--model",default="qwen38-27b-radixark-dflash2"); ap.add_argument("--out",required=True); ap.add_argument("--repeats",type=int,default=3)
    args=ap.parse_args(); groups=[]
    # one warmup, excluded
    one(args.endpoint,args.model,0,-1,0)
    for level in (1,2,3):
        for rep in range(args.repeats):
            gs=time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=level) as ex:
                futs=[ex.submit(one,args.endpoint,args.model,level,rep,i) for i in range(level)]
                rows=[f.result() for f in futs]
            ge=time.perf_counter(); wall=ge-gs
            groups.append({"level":level,"repeat":rep,"wall_s":round(wall,6),"aggregate_output_tokens_per_s":round(sum(r["completion_tokens"] for r in rows)/wall,4),"pass":all(r["pass"] for r in rows),"rows":rows})
            print(level,rep,groups[-1]["aggregate_output_tokens_per_s"],flush=True)
            time.sleep(1)
    summary={}
    for level in (1,2,3):
        gs=[g for g in groups if g["level"]==level]
        rs=[r for g in gs for r in g["rows"]]
        summary[f"c{level}"]={
            "groups":len(gs),"requests":len(rs),"pass":all(g["pass"] for g in gs),
            "aggregate_median":round(statistics.median(g["aggregate_output_tokens_per_s"] for g in gs),4),
            "per_request_decode_median":round(statistics.median(r["decode_tokens_per_s"] for r in rs if isinstance(r["decode_tokens_per_s"],float)),4),
            "ttft_median_s":round(statistics.median(r["ttft_s"] for r in rs),4),"ttft_max_s":round(max(r["ttft_s"] for r in rs),4),
        }
    result={"endpoint":args.endpoint,"model":args.model,"temperature":0,"max_tokens":512,"repeats":args.repeats,"summary":summary,"groups":groups,"pass":all(v["pass"] for v in summary.values())}
    Path(args.out).write_text(json.dumps(result,indent=2))
    print(json.dumps(summary,indent=2))
    if not result["pass"]: raise SystemExit(2)
if __name__=="__main__": main()
