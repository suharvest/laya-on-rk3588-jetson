import json,time,urllib.request,urllib.error,hashlib
out='/home/cat/kev-app/kev-perf-bins-final.jsonl'; endpoint='http://127.0.0.1:18771/api/decide'
# tokenizer is byte/BPE; repeated short words provide controlled approximate bins
bins=[('short',40),('2k',2000),('8k',7800)]
with open(out,'w') as f:
 for name,n in bins:
  for i in range(3):
   text=('alpha ' * n).strip()
   b={'input_text':text,'prompt':'Choose one option.','options':[{'id':'a','label':'Accept','description':'accept request'},{'id':'b','label':'Reject','description':'reject request'}]}
   rec={'bin':name,'target_words':n,'sample':i+1,'input_chars':len(text),'t0':time.time()}
   try:
    req=urllib.request.Request(endpoint,data=json.dumps(b).encode(),headers={'Content-Type':'application/json'}); t=time.time(); raw=urllib.request.urlopen(req,timeout=120).read(); rec.update(status=200,elapsed_ms=(time.time()-t)*1000,response=json.loads(raw))
   except urllib.error.HTTPError as e:
    rec.update(status=e.code,error=e.read().decode(errors='replace'))
   except Exception as e: rec.update(status=0,error=repr(e))
   rec['t1']=time.time(); f.write(json.dumps(rec)+'\n'); f.flush(); print(name,i+1,rec.get('status'),rec.get('elapsed_ms'),flush=True)
print('sha256',hashlib.sha256(open(out,'rb').read()).hexdigest())
