import json, re, statistics as st
P="prof/"
def load(t):
    d=json.load(open(P+"prof_%s.json"%t))
    return [(e["name"],e["averageMs"],e["medianMs"]) for e in d[1:]]
def total(t):
    for line in reversed(open(P+"prof_%s.log"%t).read().splitlines()):
        m=re.search(r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+Total\s*$",line)
        if m: return float(m.group(2))
def segment(rows):
    """group nodes into: pre, enc.layers_N (all nodes between successive Wqkv), post"""
    idx=[]
    for i,(n,a,m) in enumerate(rows):
        mm=re.search(r"/encoder/layers_(\d+)/attn/Wqkv/MatMul",n)
        if mm: idx.append((i,int(mm.group(1))))
    groups=[]
    if idx:
        groups.append(("pre(emb)",0,idx[0][0]))
        for k,(start,L) in enumerate(idx):
            end = idx[k+1][0] if k+1<len(idx) else len(rows)
            groups.append(("enc.layers_%02d"%L,start,end))
    # split trailing tail into post groups by known boundaries
    tail_start=idx[-1][0]
    for k,(s,e) in enumerate(groups):
        if s==tail_start:
            groups[k]=("enc.layers_%02d"%idx[-1][1],s,e)
    if idx:
        last_end = len(rows)
        rows_post = rows[idx[-1][0]:]
        # find start of decoder head: first node matching __myl_CasGatResAdd (intermediate) - keep whole tail as enc.last for now
        groups = groups[:-1] + [("enc.layers_%02d"%idx[-1][1], idx[-1][0], len(rows))]
    return groups

for t in ["s90_graph","s512_graph","nano.s512_graph"]:
    fn = t if t.startswith("nano") else t
    base = "prof_nano/" if t.startswith("nano") else P
    d=json.load(open(base+"prof_%s.json"%("s512_graph" if t.startswith("nano") else t)))
    rows=[(e["name"],e["averageMs"],e["medianMs"]) for e in d[1:]]
    tot=sum(r[1] for r in rows)
    idx=[(i,int(re.search(r"/encoder/layers_(\d+)/attn/Wqkv/MatMul",n).group(1))) for i,(n,a,m) in enumerate(rows) if re.search(r"/encoder/layers_(\d+)/attn/Wqkv/MatMul",n)]
    per={}
    for k,(s,L) in enumerate(idx):
        e = idx[k+1][0] if k+1<len(idx) else len(rows)
        per[L]=sum(x[1] for x in rows[s:e])
    pre=sum(rows[i][1] for i in range(0,idx[0][0]))
    print("=== %s  Total=%.4f  nodes=%d"%(t,tot,len(rows)))
    print("  pre-encoder (indices 0..%d): %.4f ms (%.2f%%)  [%s]"%(idx[0][0]-1,pre,100*pre/tot,", ".join(x[0][:38] for x in rows[:idx[0][0]])))
    print("  encoder layers: %d  per-layer min=%.4f max=%.4f mean=%.4f  SUM=%.4f (%.2f%%)"%(len(per),min(per.values()),max(per.values()),st.mean(list(per.values())),sum(per.values()),100*sum(per.values())/tot))
    print("  per-layer: "+", ".join("L%02d=%.3f"%(k,per[k]) for k in sorted(per)))
    print("  post-encoder tail: %.4f ms (%.2f%%), %d nodes starting at '%s'"%(tot-pre-sum(per.values()),100*(tot-pre-sum(per.values()))/tot,len(rows)-idx[-1][0],rows[idx[-1][0]][0][:60]))
print()
print("=== nano/nx per-node ratio (s512_graph, same engine content, 6 vs 8 SM) ===")
na={n:a for n,a,m in load("s512_graph")}
import os
nn={n:a for n,a,m in [(e["name"],e["averageMs"],e["medianMs"]) for e in json.load(open("prof_nano/prof_s512_graph.json"))[1:]]}
cats=[("attn_core_SDPA",lambda n:n.startswith("_gemm_mha_v2")),("attn_qkv_gemm",lambda n:"/attn/Wqkv/MatMul" in n),
      ("attn_out_proj",lambda n:"/attn/Wo/MatMul" in n),("ffn_gemm",lambda n:"/mlp/Wo/MatMul" in n or "/linear1/MatMul" in n),
      ("ffn_fused_fc",lambda n:bool(re.match(r"__myl_Fc",n))),("norm_residual_ptw",lambda n:"AddCasMea" in n),
      ("rotary_apply",lambda n:"TraSliSli" in n or "TraConSinCos" in n)]
for c,f in cats:
    ks=[k for k in na if f(k) and k in nn]
    if not ks: continue
    ra=[nn[k]/na[k] for k in ks]
    print("  %-20s n=%3d  nano/nx median=%.3f  min=%.3f max=%.3f   (nx sum=%.4f nano sum=%.4f)"%(c,len(ks),st.median(ra),min(ra),max(ra),sum(na[k] for k in ks),sum(nn[k] for k in ks)))
