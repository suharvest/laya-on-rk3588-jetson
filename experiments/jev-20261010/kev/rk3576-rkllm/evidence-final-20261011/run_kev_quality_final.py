import json, time, urllib.request, urllib.error, hashlib, os
base='http://127.0.0.1:18771/api/decide'
source='/home/cat/kev-app/jev-shared-cases-20261010.jsonl'
out='/home/cat/kev-app/kev-quality-final-240.raw.jsonl'
rows=[json.loads(x) for x in open(source) if x.strip()]
start=time.time(); ok=correct=errors=overflows=0; lat=[]
with open(out,'w') as f:
  for pass_name, reverse in [('orig',False),('reverse',True)]:
    for i,x in enumerate(rows):
      opts=x['options'][::-1] if reverse else x['options']
      body={'input_text':x['input'],'prompt':x['prompt'],'options':[{'id':o['id'],'label':o['title'],'description':o.get('description','')} for o in opts]}
      rec={'pass':pass_name,'id':x['id'],'expected_id':x['expected_id'],'t0':time.time()}
      req=urllib.request.Request(base,data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
      try:
        t=time.time(); data=json.loads(urllib.request.urlopen(req,timeout=90).read()); ms=(time.time()-t)*1000
        rec.update(response=data,elapsed_ms=ms,status=200)
        ok+=1; correct+=int(data.get('selected_option_id')==x['expected_id']); lat.append(ms)
      except urllib.error.HTTPError as e:
        raw=e.read().decode(errors='replace'); rec.update(status=e.code,error=raw)
        errors+=1; overflows+=int(e.code==422 and 'input_too_long' in raw)
      except Exception as e:
        rec.update(status=0,error=repr(e)); errors+=1
      rec['t1']=time.time(); f.write(json.dumps(rec,ensure_ascii=False)+'\n'); f.flush()
      n=ok+errors
      if n%10==0: print(f'PROGRESS n={n}/240 ok={ok} correct={correct} errors={errors} elapsed={time.time()-start:.1f}s',flush=True)
summary={'total':240,'ok':ok,'correct':correct,'accuracy':correct/ok if ok else None,'errors':errors,'token_overflow_422':overflows,'elapsed_s':time.time()-start,'latency_ms_mean':sum(lat)/len(lat) if lat else None,'latency_ms_p95':sorted(lat)[int(.95*len(lat))-1] if lat else None,'raw':out,'source_sha256':hashlib.sha256(open(source,'rb').read()).hexdigest()}
sp=out.replace('.raw.jsonl','.summary.json'); open(sp,'w').write(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)
