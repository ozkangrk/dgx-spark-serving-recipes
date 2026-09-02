#!/usr/bin/env python3
import json, urllib.request, uuid
from pathlib import Path
URL='http://127.0.0.1:8888/v1/chat/completions'; MODEL='qwen38-27b-radixark-dflash2'

def call(payload):
    req=urllib.request.Request(URL,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer local-key'})
    with urllib.request.urlopen(req,timeout=120) as r: return json.load(r)

rows=[]
for i in range(10):
    q=f'Nonce {uuid.uuid4().hex}. Compute 17*23. Return only the integer.'
    r=call({'model':MODEL,'messages':[{'role':'user','content':q}],'temperature':0,'max_tokens':32,'chat_template_kwargs':{'enable_thinking':False}})
    text=(r['choices'][0]['message'].get('content') or '').strip()
    rows.append({'kind':'arithmetic','i':i,'text':text,'pass':text=='391','usage':r.get('usage')})

tool={'type':'function','function':{'name':'get_weather','description':'Get weather','parameters':{'type':'object','properties':{'city':{'type':'string'},'unit':{'type':'string','enum':['celsius','fahrenheit']}},'required':['city','unit'],'additionalProperties':False}}}
for i in range(10):
    q=f'Nonce {uuid.uuid4().hex}. Use get_weather for Istanbul in celsius.'
    r=call({'model':MODEL,'messages':[{'role':'user','content':q}],'temperature':0,'max_tokens':128,'tools':[tool],'tool_choice':{'type':'function','function':{'name':'get_weather'}},'chat_template_kwargs':{'enable_thinking':False}})
    calls=r['choices'][0]['message'].get('tool_calls') or []
    ok=False; args=None
    if len(calls)==1 and calls[0].get('function',{}).get('name')=='get_weather':
        try:
            args=json.loads(calls[0]['function']['arguments']); ok=args=={'city':'Istanbul','unit':'celsius'}
        except Exception: pass
    rows.append({'kind':'tool','i':i,'args':args,'call_count':len(calls),'pass':ok,'usage':r.get('usage')})
out={'model':MODEL,'total':len(rows),'passed':sum(x['pass'] for x in rows),'pass':all(x['pass'] for x in rows),'rows':rows}
Path(__file__).resolve().parent/('candidate-quality-smoke.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
raise SystemExit(0 if out['pass'] else 2)
