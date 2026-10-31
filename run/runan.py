import json,sys,itertools,random,statistics
sys.path.insert(0,"src")
from g4paper import analysis as A
S=json.load(open("sample_ids.json"))
lab={(x["submission"],x["instance_id"]):int(bool(x["resolved"])) for x in S}
rows={}
for f in ["run/judgments_nebius.jsonl","run/judgments_ibm.jsonl","run/judgments_gemma4.jsonl"]:
    for l in open(f):
        r=json.loads(l); k=(r["submission"],r["instance_id"],r["judge"])
        if k[:2] not in lab: continue
        if k not in rows or (rows[k].get("p") is None and r.get("p") is not None): rows[k]=r
R=list(rows.values())
MAKER={"gpt-oss-120b":"OpenAI","qwen3-235b":"Alibaba","deepseek-v4-flash":"DeepSeek","glm-5.2":"Zhipu","gemma-3-27b":"Google","gemma-4-e4b":"Google","gemma-4-e2b":"Google",
 "ibm-granite-4-small":"IBM","ibm-llama4-maverick":"Meta","ibm-mistral-small":"Mistral"}
J=sorted({r["judge"] for r in R})
out={"n_items":len(lab),"resolved":sum(lab.values()),"judges":{}}
pj=A.per_judge(R,lab)
rng=random.Random(2026)
def boot(judge):
    pts=[(r["p"],lab[(r["submission"],r["instance_id"])]) for r in R if r["judge"]==judge and r.get("p") is not None]
    v=[]
    for _ in range(500):
        s=[pts[rng.randrange(len(pts))] for _ in pts]; a=A.auroc([x for x,_ in s],[y for _,y in s])
        if a is not None: v.append(a)
    v.sort(); return [round(v[int(.025*len(v))],3),round(v[int(.975*len(v))],3)]
for j in J:
    d=dict(pj[j]); d["maker"]=MAKER.get(j,"?"); d["auroc_ci95"]=boot(j); out["judges"][j]={k:(round(v,4) if isinstance(v,float) else v) for k,v in d.items()}
def ens(js):
    e=A.ensemble(R,lab,list(js)); return {k:(round(v,4) if isinstance(v,float) else v) for k,v in e.items()}
out["ensemble_all"]=ens(J)
same=["gemma-3-27b","gemma-4-e4b","gemma-4-e2b"]
out["same_maker_google_triple"]=ens(same)
cross=[c for c in itertools.combinations(J,3) if len({MAKER[x] for x in c})==3]
vals=[]
for c in cross:
    e=A.ensemble(R,lab,list(c)); a=e.get("auroc")
    if a is not None: vals.append((a,c,e.get("n")))
vals.sort(reverse=True)
out["cross_maker_triples"]={"count":len(vals),"auroc_median":round(statistics.median(v[0] for v in vals),4),"auroc_min":round(vals[-1][0],4),"auroc_max":round(vals[0][0],4),"best":[vals[0][1],vals[0][2]],
  "share_beating_same_maker":round(sum(1 for v in vals if v[0]>out["same_maker_google_triple"]["auroc"])/len(vals),3)}
g4=["gemma-4-e4b","gemma-4-e2b"]; best3=list(vals[0][1])
out["agreement_curve_best_cross_triple"]=A.agreement_curve(R,lab,best3)
out["agreement_curve_all"]=A.agreement_curve(R,lab,J)
out["gemma4_added_to_best_cross"]=ens(best3+["gemma-4-e4b"])
json.dump(out,open("run/results_analysis.json","w"),indent=1,default=str)
print(json.dumps(out,default=str)[:6000])
