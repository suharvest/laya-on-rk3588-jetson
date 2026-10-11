import json,math,hashlib,os
BASE="/mnt/f/bekko-rknn-20261010"; E=BASE+"/evidence/"
def wilson(k,n,z=1.959963984540054):
 p=k/n; den=1+z*z/n; ctr=(p+z*z/(2*n))/den; rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return [ctr-rad,ctr+rad]
def load(f): return {x["id"]:x for x in json.load(open(E+f))}
files={"browser_v3":"browser-quality-120-v3.json","browser_reverse_v3":"browser-quality-120-reverse-v3.json","browser_fixed_v3":"browser-quality-120-fixed-v3.json","native_fp32":"native-quality-120.json"}; ds={x["id"]:x for x in map(json.loads,open(BASE+"/jev-shared-cases-20261010.jsonl"))}; rows={k:load(v) for k,v in files.items()}; join=[]
for cid,c in ds.items():
 r={"id":cid,"language":c["language"],"group":c.get("group"),"category":c.get("category"),"scenario":c.get("scenario"),"expected_id":c["expected_id"]}
 for k,m in rows.items(): r[k]={"selected_id":m[cid].get("selected_id"),"correct":m[cid].get("correct"),"error":m[cid].get("error")}
 join.append(r)
def stats(k,subset):
 a=[x for x in join if subset(x)]; good=sum(x[k]["correct"] is True for x in a); n=len(a); return {"n":n,"correct":good,"accuracy":good/n if n else None,"wilson95":wilson(good,n) if n else None}
summary={"dataset_sha256":hashlib.sha256(open(BASE+"/jev-shared-cases-20261010.jsonl","rb").read()).hexdigest(),"denominator":len(join),"variants":files,"overall":{k:stats(k,lambda x:True) for k in rows},"by_language":{lang:{k:stats(k,lambda x,lang=lang:x["language"]==lang) for k in rows} for lang in ["en","zh"]},"by_category":{cat:{k:stats(k,lambda x,cat=cat:x["category"]==cat) for k in rows} for cat in sorted(set(x["category"] for x in join))},"join":join}
json.dump(summary,open(E+"browser-v3-summary-joined.json","w"),ensure_ascii=False,indent=2)
prov={"source_model":"hotchpotch/bekko-system-one-v0-68m","source_revision":"ab7685f23e5edbc1acb12ced4f2c4e12591efa69","model_license":"HF README says released-weight and bundled inference-code license declaration remains to be finalized","browser_onnx_sha256":"3b121d68a67f9678a1272a08f24dd2ccffb276f92f39a3f77953b30290cbc3e1","browser_tokenizer_sha256":"dd8374bb8bd74c2112989a24697e0ba494c6258ed758c0701a62957e67d38cea","native_weights_sha256":"db2f01fd59d5a0b38d14f8befd981a33db6d462cd652980cf9e29b0798aca71c","native_fp32_onnx_path":"/home/harve/tmp/bekko-native-20261010/native-fixed128-384-fp32-cpu.onnx","native_fp32_onnx_sha256":"42e69d0d6eb1a0fb96d99d0483bb8d4e2e43fbb45d307164c6978ac15cf93ef0","native_ort_smoke":{"case":"jev-001-en","max_abs_logit_error":0.05402183532714844,"mean_abs_logit_error":0.03401700779795647,"finite":True},"codec_source":"https://github.com/hotchpotch/bekko-system-one/blob/main/browser/src/core.js","head_source":"manifest tasks [choice,noul,score]; predict column=tasks.indexOf(request.task), choice index 0","dataset_sha256":summary["dataset_sha256"]}
json.dump(prov,open(E+"browser-v3-provenance.json","w"),ensure_ascii=False,indent=2)
print(json.dumps(summary["overall"],ensure_ascii=False))
