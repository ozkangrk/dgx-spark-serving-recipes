#!/usr/bin/env python3
import json, random, statistics, time, urllib.request, uuid
from pathlib import Path
URL='http://127.0.0.1:8888/v1/chat/completions'; MODEL='qwen38-27b-radixark-dflash2'
WORDS=('alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu scheduler kernel latency throughput quantize tensor gradient cache pointer buffer register pipeline parallel entropy manifold').split()

def prompt(seed, approx_tokens, fixed_nonce=None):
 rng=random.Random(seed); n=max(1,int(approx_tokens/1.3)); body=' '.join(rng.choice(WORDS) for _ in range(n))
 nonce=fixed_nonce or uuid.uuid4().hex
 return f'UNIQUE_NONCE_{nonce}\nReference log:\n{body}\nEnd log. Implement a complete Python LRUCache using OrderedDict with type hints and concise unit tests. Output only Python code and continue with edge-case tests until the token limit.'

def run(text):
 payload={'model':MODEL,'messages':[{'role':'user','content':text}],'temperature':0,'max_tokens':512,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
 req=urllib.request.Request(URL,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer local-key'})
 t0=time.perf_counter(); first=None; usage=None; finish=None
 with urllib.request.urlopen(req,timeout=600) as resp:
  for raw in resp:
   line=raw.decode().rstrip('\r\n')
   if not line.startswith('data:'): continue
   s=line[5:].lstrip()
   if s=='[DONE]': break
   obj=json.loads(s)
   if obj.get('usage') is not None: usage=obj['usage']
   ch=obj.get('choices') or []
   if ch:
    finish=ch[0].get('finish_reason') or finish
    x=(ch[0].get('delta') or {}).get('content')
    if isinstance(x,str) and x: first=first or time.perf_counter()
 t1=time.perf_counter()
 if not usage or first is None: raise RuntimeError('missing usage/ttft')
 pt=usage['prompt_tokens']; ct=usage['completion_tokens']; dw=t1-first
 return {'prompt_tokens':pt,'completion_tokens':ct,'finish_reason':finish,'ttft_s':round(first-t0,4),'prefill_tok_s':round(pt/(first-t0),2),'decode_tok_s':round((ct-1)/dw,2),'e2e_s':round(t1-t0,4)}

# excluded warmup
run(prompt(0,320))
rows=[]
for depth in (320,8000,65000):
 vals=[]
 for i in range(4):
  r=run(prompt(depth*100+i,depth)); r['target_depth']=depth; r['repeat']=i; vals.append(r); rows.append(r); print(depth,i,r,flush=True)
# Exact repeat-prefix pair, separate from cold rows.
shared=prompt(999999,65000,fixed_nonce='CACHE_PAIR_FIXED')
cache_first=run(shared); cache_second=run(shared)
summary={}
for depth in (320,8000,65000):
 v=[r for r in rows if r['target_depth']==depth]
 summary[str(depth)]={'prompt_tokens_median':statistics.median(r['prompt_tokens'] for r in v),'decode_tok_s_median':round(statistics.median(r['decode_tok_s'] for r in v),2),'ttft_s_median':round(statistics.median(r['ttft_s'] for r in v),3),'prefill_tok_s_median':round(statistics.median(r['prefill_tok_s'] for r in v),2),'all_512':all(r['completion_tokens']==512 for r in v)}
out={'model':MODEL,'repeats':4,'summary':summary,'cache_pair_65k':{'first':cache_first,'second':cache_second},'rows':rows}
Path(__file__).resolve().parent/('candidate-context-ladder.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'summary':summary,'cache_pair_65k':out['cache_pair_65k']},indent=2))
