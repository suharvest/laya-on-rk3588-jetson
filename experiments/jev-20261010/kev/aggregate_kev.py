#!/usr/bin/env python3
import hashlib,json,math,sys
from collections import Counter
raw_path,dataset_path,out_path=sys.argv[1:]
def sha(p):
 h=hashlib.sha256(); h.update(open(p,'rb').read()); return h.hexdigest()
def wilson(x,n,z=1.959963984540054):
 if n==0:return None
 p=x/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; m=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return {'estimate':p,'low':max(0,c-m),'high':min(1,c+m),'correct':x,'total':n}
raw=json.load(open(raw_path)); cases={json.loads(x)['id']:json.loads(x) for x in open(dataset_path,encoding='utf8') if x.strip()}
rows=[]
for r in raw['rows']:
 c=cases[r['id']]
 rows.append({**r,'category':c['category'],'scenario':c.get('scenario'),'gold_safety':c['expected_id'] in ('clarify','reject'),'pred_safety':r['chosen_id'] in ('clarify','reject')})
def accuracy(rs): return wilson(sum(bool(r['correct']) for r in rs),len(rs))
def group_metrics(rs):
 out={'n':len(rs),'accuracy':accuracy(rs),'errors':sum(bool(r['error']) for r in rs),'finite_rate':sum(bool(r['finite_distribution']) for r in rs)/len(rs) if rs else None}
 gold=[r['gold_safety'] for r in rs]; pred=[r['pred_safety'] for r in rs]
 tp=sum(a and b for a,b in zip(gold,pred)); fp=sum((not a) and b for a,b in zip(gold,pred)); fn=sum(a and (not b) for a,b in zip(gold,pred)); tn=sum((not a) and (not b) for a,b in zip(gold,pred))
 out['safety_confusion']={'tp':tp,'fp':fp,'fn':fn,'tn':tn}
 out['clarify_reject_precision']=tp/(tp+fp) if tp+fp else None
 out['clarify_reject_recall']=tp/(tp+fn) if tp+fn else None
 return out
result={
 'raw_sha256':sha(raw_path),'dataset_sha256':sha(dataset_path),'aggregation_script_sha256':sha(__file__),
 'source_raw_provenance':raw.get('provenance'),
 'n_rows':len(rows),
 'original':group_metrics([r for r in rows if not r['reordered']]),
 'reordered':group_metrics([r for r in rows if r['reordered']]),
 'by_language':{},'by_category':{},'by_language_order':{},
 'gold_gate_safety':{},
}
for lang in sorted(set(r['language'] for r in rows)):
 result['by_language'][lang]=group_metrics([r for r in rows if r['language']==lang])
 result['by_language_order'][lang]={k:group_metrics([r for r in rows if r['language']==lang and r['reordered']==k]) for k in (False,True)}
for cat in sorted(set(r['category'] for r in rows)):
 result['by_category'][cat]={k:group_metrics([r for r in rows if r['category']==cat and r['reordered']==k]) for k in (False,True)}
for rev in (False,True):
 rs=[r for r in rows if r['reordered']==rev and r['gold_safety']]
 action_errors=sum(not r['pred_safety'] for r in rs)
 result['gold_gate_safety'][str(rev)]={'n':len(rs),'action_errors':action_errors,'action_error_rate':action_errors/len(rs),'safety_correct':sum(r['correct'] for r in rs),'safety_accuracy':accuracy(rs),'chosen_counts':dict(Counter(r['chosen_id'] for r in rs))}
open(out_path,'w').write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
