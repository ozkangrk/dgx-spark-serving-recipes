#!/usr/bin/env python3
import concurrent.futures, json, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent/('hermes-c123-v2')
ROOT.mkdir(parents=True,exist_ok=True)
TASKS=[
 ('quantdeveloper','A','Create result.json in the exact current workdir containing exactly this JSON object: {"task":"A","product":391}. Use write_file, read it back, then finish.'),
 ('coder','B','Read input.txt in the exact current workdir. Create result.json containing exactly {"task":"B","sum":15,"count":5}. Verify it, then finish.'),
 ('tester','C','Fix calc.py in the exact current workdir so subtract(10,3) returns 7. Run test_calc.py and create result.json containing exactly {"task":"C","tests_passed":true}. Then finish.'),
]

def prep(group, tid):
 d=ROOT/group/tid; d.mkdir(parents=True,exist_ok=True)
 if tid=='B': (d/'input.txt').write_text('1\n2\n3\n4\n5\n')
 if tid=='C':
  (d/'calc.py').write_text('def subtract(a,b):\n    return a+b\n')
  (d/'test_calc.py').write_text("from calc import subtract\nassert subtract(10,3)==7\nprint('PASS')\n")
 return d

def run(group, item):
 profile,tid,prompt=item; d=prep(group,tid)
 prompt = prompt + f' The exact benchmark workdir is {d}. Every file read/write and terminal command must use this absolute directory; do not write outside this workdir.'
 cmd=['hermes','--profile',profile,'--in',str(d),'chat','-q',prompt,'--max-turns','8']
 t=time.monotonic(); p=subprocess.run(cmd,text=True,capture_output=True,timeout=300); wall=time.monotonic()-t
 ok=False; got=None; err=None
 try:
  got=json.loads((d/'result.json').read_text())
  expected={'A':{'task':'A','product':391},'B':{'task':'B','sum':15,'count':5},'C':{'task':'C','tests_passed':True}}[tid]
  ok=(got==expected)
  if tid=='C': ok=ok and subprocess.run(['python3','test_calc.py'],cwd=d,capture_output=True).returncode==0
 except Exception as e: err=f'{type(e).__name__}: {e}'
 return {'group':group,'profile':profile,'task':tid,'wall_s':round(wall,3),'exit_code':p.returncode,'pass':ok,'result':got,'error':err,'stdout_tail':p.stdout[-1000:],'stderr_tail':p.stderr[-1000:]}

all_groups=[]
for c in (1,2,3):
 group=f'C{c}'; start=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
  rows=list(ex.map(lambda x:run(group,x),TASKS[:c]))
 wall=time.monotonic()-start
 rec={'group':group,'wall_s':round(wall,3),'pass':all(r['pass'] for r in rows),'rows':rows}
 all_groups.append(rec); print(json.dumps({'group':group,'wall_s':rec['wall_s'],'pass':rec['pass'],'task_walls':[r['wall_s'] for r in rows]}),flush=True)
out={'groups':all_groups,'pass':all(g['pass'] for g in all_groups)}
(ROOT.parent/'candidate-hermes-c123-v2.json').write_text(json.dumps(out,indent=2))
raise SystemExit(0 if out['pass'] else 2)
