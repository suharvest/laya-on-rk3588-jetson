#!/usr/bin/env python3
import importlib.util, json, math, os, random, subprocess, sys, time, hashlib
from pathlib import Path
import numpy as np

ROOT = Path('/home/harvest/tmp/jev-20261010/laya')
SRC = ROOT/'native.py'
DATA = ROOT/'jev-shared-cases-20261010.jsonl'
TOK = ROOT/'tokenizer'
ENGINE = ROOT/'laya.s512.fp16.engine'
OUT = ROOT/'laya512-baseline-reversed-20261010.json'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def mem_avail():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1])*1024
    return None
def gpu_query():
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=5)
        return {'rc':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
    except Exception as e: return {'error':str(e)}
def load_native():
    spec=importlib.util.spec_from_file_location('laya_native',SRC)
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    return mod
def options_for(row, reverse=False, rng=None):
    opts=[{'id':str(o['id']),'title':o['title'],'description':o.get('description','')} for o in row['options']]
    if reverse:
        rng.shuffle(opts)
    return [{'id':o['id'],'label':o['title'],'description':o['description']} for o in opts]
def run_pass(name, rows, codec, backend, reverse=False):
    rng=random.Random(20261010)
    records=[]; errors=[]
    for row in rows:
        rid=str(row['id']); opts=options_for(row,False,rng)
        if reverse: opts.reverse()
        try:
            req=codec.encode(str(row['prompt']),str(row.get('input','')),opts)
            scores=backend.infer_request(req)
            finite=bool(np.isfinite(np.asarray(scores,dtype=np.float64)).all())
            if not finite: raise RuntimeError('non-finite active logits')
            order=sorted(range(len(scores)), key=lambda i:(-float(scores[i]),i))
            selected=req.option_ids[order[0]]
            expected=str(row['expected_id'])
            records.append({'id':rid,'language':row.get('language'),'category':row.get('category'),'scenario':row.get('scenario'),'selected_id':selected,'expected_id':expected,'correct':selected==expected,'scores':{oid:float(s) for oid,s in zip(req.option_ids,scores)},'ranking':[req.option_ids[i] for i in order],'tokens':int(req.attention_mask.sum()),'finite':finite,'model_visible':{'prompt':str(row['prompt']),'input':str(row.get('input','')),'options':opts},'option_order':[o['id'] for o in opts]})
        except Exception as e:
            errors.append({'id':rid,'error':f'{type(e).__name__}: {e}'})
            records.append({'id':rid,'language':row.get('language'),'category':row.get('category'),'selected_id':None,'expected_id':str(row['expected_id']),'correct':False,'error':f'{type(e).__name__}: {e}'})
    return {'name':name,'n':len(rows),'records':records,'errors':errors}
def metrics(passdata):
    rs=passdata['records']; n=len(rs); acc=sum(bool(r.get('correct')) for r in rs)
    def group(key):
        out={}
        for r in rs:
            k=str(r.get(key)); x=out.setdefault(k,{'correct':0,'n':0,'errors':[]}); x['n']+=1; x['correct']+=int(bool(r.get('correct')))
            if 'error' in r: x['errors'].append(r['id'])
        for x in out.values(): x['accuracy']=x['correct']/x['n'] if x['n'] else None
        return out
    return {'correct':acc,'n':n,'accuracy':acc/n if n else None,'errors':passdata['errors'],'by_language':group('language'),'by_category':group('category'),'by_scenario':group('scenario')}
def main():
    pre_mem=mem_avail(); pre_gpu=gpu_query()
    if pre_mem is not None and pre_mem < 2.5*1024**3: raise RuntimeError(f'MemAvailable below gate: {pre_mem}')
    rows=[json.loads(x) for x in DATA.read_text().splitlines() if x.strip()]
    if len(rows)!=120: raise RuntimeError(f'expected 120 rows, got {len(rows)}')
    mod=load_native(); codec=mod.LayaCodec(str(TOK),512,16); backend=mod.TensorRTBackend(str(ENGINE),512,16)
    if not backend.ready: raise RuntimeError(f'backend unavailable: {backend.reason}')
    original=run_pass('original',rows,codec,backend,False)
    reversed_pass=run_pass('deterministic_reversed_options',rows,codec,backend,True)
    # Timing one representative frozen encoded request, inference only; backend synchronizes each call.
    req=codec.encode(str(rows[0]['prompt']),str(rows[0].get('input','')),options_for(rows[0],False,random.Random(20261010)))
    for _ in range(20): backend.infer_request(req)
    lat=[]
    for _ in range(50):
        t=time.perf_counter(); backend.infer_request(req); lat.append((time.perf_counter()-t)*1000)
    sl=sorted(lat); p95=sl[math.ceil(.95*len(sl))-1]
    post_mem=mem_avail(); post_gpu=gpu_query()
    byid={r['id']:r for r in original['records']}; perm={r['id']:r for r in reversed_pass['records']}
    invariance={'same_selected':sum(byid[k].get('selected_id')==perm[k].get('selected_id') for k in byid),'n':len(byid),'changed_ids':[k for k in byid if byid[k].get('selected_id')!=perm[k].get('selected_id')]}
    result={'schema':'jev-laya512-baseline-20261010.v1','dataset_sha256':sha(DATA),'dataset_n':len(rows),'source_sha256':sha(SRC),'tokenizer_sha256':sha(TOK/'tokenizer.json'),'tokenizer_config_sha256':sha(TOK/'tokenizer_config.json'),'engine_sha256':sha(ENGINE),'engine_bytes':ENGINE.stat().st_size,'config':{'seq_len':512,'max_options':16,'backend':'TensorRT','runtime':backend.runtime_version,'warmup':20,'timed_iters':50,'timing_scope':'backend.infer_request; native backend synchronizes CUDA','p95':'nearest-rank'},'preflight':{'mem_available_bytes':pre_mem,'gpu':pre_gpu},'during_after':{'mem_available_bytes':post_mem,'gpu':post_gpu},'passes':{'original':{'metrics':metrics(original),'records':original['records']},'deterministic_reversed_options':{'metrics':metrics(reversed_pass),'records':reversed_pass['records']}},'permutation_invariance':invariance,'timing_ms':{'min':min(lat),'median':sl[len(sl)//2],'p95_nearest_rank':p95,'max':max(lat),'samples':lat}}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'out':str(OUT),'original':metrics(original),'permutation':metrics(reversed_pass),'invariance':invariance,'timing_ms':result['timing_ms'],'mem_before':pre_mem,'mem_after':post_mem},ensure_ascii=False))
    backend.release()
if __name__=='__main__': main()
