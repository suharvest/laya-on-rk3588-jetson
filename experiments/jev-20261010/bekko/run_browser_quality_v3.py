import json,time,hashlib,sys
import numpy as np
from tokenizers import Tokenizer
import onnxruntime as ort
BASE="/mnt/f/bekko-rknn-20261010"; reverse=(len(sys.argv)>1 and sys.argv[1]=="reverse")
manifest=json.load(open(BASE+"/browser/manifest.json")); tok=Tokenizer.from_file(BASE+"/browser/tokenizer.json"); sess=ort.InferenceSession(BASE+"/model.onnx",providers=["CPUExecutionProvider"])
model_sha=hashlib.sha256(open(BASE+"/model.onnx","rb").read()).hexdigest(); tok_sha=hashlib.sha256(open(BASE+"/browser/tokenizer.json","rb").read()).hexdigest(); code_sha=hashlib.sha256(open(__file__,"rb").read()).hexdigest()
cases=[json.loads(x) for x in open(BASE+"/jev-shared-cases-20261010.jsonl") if x.strip()]; assert len(cases)==120
def enc(s): return tok.encode(s,add_special_tokens=False).ids
def jsdump(x): return json.dumps(x,ensure_ascii=False,separators=(",",":"))
def make(case):
 opts=list(case["options"]); opts=opts[::-1] if reverse else opts; instruction=case["prompt"]; state=jsdump({"input":case["input"]}); desc=[{"id":o["title"],"description":o["description"]} for o in opts]
 im="Instruction: "; sm="State: "; sep="\n"; ii=enc(instruction); si=enc(state); imi=enc(im); smi=enc(sm); sepi=enc(sep); budget=manifest["query_length"]-2-len(imi)-len(smi)-len(sepi); ni=min(len(ii),(budget+1)//2); ns=min(len(si),budget//2); ni+=min(len(ii)-ni,budget-ni-ns); ns+=min(len(si)-ns,budget-ni-ns); prefix=[manifest["cls_token_id"]]+imi+ii[:ni]+sepi+smi+si[:ns]+[manifest["sep_token_id"]]; docs=[]; doc_text=[]
 for c in desc:
  txt="Candidate: "+c["id"]+": "+c["description"]; doc_text.append(txt); docs.append(enc(txt)[:manifest["document_length"]-1]+[manifest["sep_token_id"]])
 pw=len(prefix); dw=max(len(x) for x in docs); pi=np.full((1,pw),manifest["pad_token_id"],np.int64); pm=np.zeros((1,pw),np.bool_); pi[0,:pw]=prefix; pm[0,:pw]=1; di=np.full((len(docs),dw),manifest["pad_token_id"],np.int64); dm=np.zeros((len(docs),dw),np.bool_)
 for i,row in enumerate(docs): di[i,:len(row)]=row; dm[i,:len(row)]=1
 return opts,prefix,docs,doc_text,{"prefix_ids":pi,"prefix_mask":pm,"doc_ids":di,"doc_mask":dm,"owners":np.zeros(len(docs),np.int64)},instruction,state,im,sm,sep
out=[]; correct=0; errors=0
for case in cases:
 t0=time.perf_counter()
 try:
  opts,prefix,docs,doc_text,feeds,instruction,state,im,sm,sep=make(case); t1=time.perf_counter(); rr=sess.run(["logits"],feeds)[0]; t2=time.perf_counter(); logits=np.asarray(rr,np.float32)[:,manifest["tasks"].index("choice")]; finite=bool(np.isfinite(logits).all());
  if logits.shape[0]!=len(opts) or not finite: raise ValueError("invalid logits shape/finite")
  z=logits.astype(np.float64); ex=np.exp(z-z.max()); probs=ex/ex.sum(); idx=int(np.argmax(logits)); selected=opts[idx]["id"]; t3=time.perf_counter(); ok=selected==case["expected_id"]; correct+=int(ok)
  out.append({"id":case["id"],"variant":"reverse" if reverse else "original","language":case["language"],"selected_id":selected,"expected_id":case["expected_id"],"correct":ok,"finite_active_logits":finite,"option_ids":[x["id"] for x in opts],"model_visible_input":{"instruction":instruction,"state":state,"prefix_text":im+instruction+sep+sm+state,"candidate_texts":doc_text},"input_shapes":{k:list(v.shape) for k,v in feeds.items()},"token_lengths":{"prefix":len(prefix),"documents":[len(x) for x in docs]},"logits":logits.tolist(),"probabilities":probs.tolist(),"timing_ms":{"codec":(t1-t0)*1000,"inference":(t2-t1)*1000,"decode":(t3-t2)*1000,"end_to_end":(t3-t0)*1000},"model_sha256":model_sha,"tokenizer_sha256":tok_sha,"codec_code_sha256":code_sha})
 except Exception as e: errors+=1; out.append({"id":case.get("id"),"variant":"reverse" if reverse else "original","language":case.get("language"),"correct":False,"error":type(e).__name__+": "+str(e),"model_sha256":model_sha,"tokenizer_sha256":tok_sha,"codec_code_sha256":code_sha})
name="browser-quality-120-reverse-v3.json" if reverse else "browser-quality-120-v3.json"; json.dump(out,open(BASE+"/evidence/"+name,"w"),ensure_ascii=False,indent=2); print(json.dumps({"file":name,"cases":len(cases),"correct":correct,"errors":errors,"accuracy":correct/len(cases),"variant":"reverse" if reverse else "original","model_sha256":model_sha,"tokenizer_sha256":tok_sha,"codec_code_sha256":code_sha,"ort":ort.__version__},ensure_ascii=False))
