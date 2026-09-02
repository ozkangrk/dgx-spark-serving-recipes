#!/usr/bin/env python3
import json, os, subprocess, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
NAME="${NAME:-qwen38-sglang}"
OUT=Path(__file__).resolve().parent/(datetime.now(timezone.utc).strftime("stability-%Y%m%dT%H%M%SZ.json"))
rows=[]
for i in range(12):
    row={"i":i+1,"time_utc":datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen("http://127.0.0.1:%s/health" % os.environ.get("GB10_PORT","8888"),timeout=5) as r: row["health_status"]=r.status
    except Exception as e: row["health_error"]=repr(e)
    try:
        x=subprocess.check_output(["docker","inspect",NAME,"--format",'{{json .State}}'],text=True); s=json.loads(x)
        row.update({"running":s.get("Running"),"oom_killed":s.get("OOMKilled"),"exit_code":s.get("ExitCode")})
        rc=subprocess.check_output(["docker","inspect",NAME,"--format",'{{.RestartCount}}'],text=True).strip(); row["restart_count"]=int(rc)
    except Exception as e: row["docker_error"]=repr(e)
    try:
        g=subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,power.draw,temperature.gpu","--format=csv,noheader,nounits"],text=True).strip().split(",")
        row.update({"gpu_util_pct":float(g[0]),"power_w":float(g[1]),"temp_c":float(g[2])})
    except Exception as e: row["gpu_error"]=repr(e)
    rows.append(row); print(json.dumps(row),flush=True)
    if i<11: time.sleep(15)
verdict=all(r.get("health_status")==200 and r.get("running") and not r.get("oom_killed") and r.get("restart_count")==0 for r in rows)
OUT.write_text(json.dumps({"samples":rows,"passed":verdict},indent=2)+"\n")
print(json.dumps({"output":str(OUT),"passed":verdict,"samples":len(rows)},indent=2))
raise SystemExit(0 if verdict else 2)
