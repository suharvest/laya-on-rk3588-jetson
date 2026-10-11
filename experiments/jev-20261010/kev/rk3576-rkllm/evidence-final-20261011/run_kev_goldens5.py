import json,time,urllib.request,urllib.error,hashlib
src='/home/cat/kev-app/jev-shared-cases-20261010.jsonl'; ids={'jev-001-en','jev-001-zh','jev-002-en','jev-002-zh','jev-003-en'}
out='/home/cat/kev-app/kev-goldens5-2048.raw.jsonl'
rows=[json.loads(x) for x in open(src) if x.strip() and json.loads(x)['id'] in ids]
with open(out,'w') as f:
 for x in rows:
  b={'input_text':x['input'],'prompt':x['prompt'],'options':[{'id':o['id'],'label':o['title'],'description':o.get('description','')} for o in x['options']]}
  rec={'id':x['id'],'expected_id':x['expected_id']}; t=time.time()
  try:
   req=urllib.request.Request('http://127.0.0.1:18771/api/decide',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'}); z=urllib.request.urlopen(req,timeout=90); rec.update(status=z.status,elapsed_ms=(time.time()-t)*1000,response=json.loads(z.read()))
  except urllib.error.HTTPError as e: rec.update(status=e.code,error=e.read().decode())
  except Exception as e: rec.update(status=0,error=repr(e))
  f.write(json.dumps(rec,ensure_ascii=False)+'\n'); f.flush(); print(rec['id'],rec['status'],rec.get('elapsed_ms'),flush=True)
print('sha256',hashlib.sha256(open(out,'rb').read()).hexdigest())
