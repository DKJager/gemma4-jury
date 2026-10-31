import json,sys
sys.path.insert(0,"src")
from g4paper import analysis as A
S=json.load(open("sample_ids.json")); lab={(x["submission"],x["instance_id"]):int(bool(x["resolved"])) for x in S}
rows={}
for f in ["run/judgments_nebius.jsonl","run/judgments_ibm.jsonl","run/judgments_gemma4.jsonl"]:
    for l in open(f):
        r=json.loads(l); k=(r["submission"],r["instance_id"],r["judge"])
        if k[:2] in lab and (k not in rows or (rows[k].get("p") is None and r.get("p") is not None)): rows[k]=r
R=list(rows.values()); J=sorted({r["judge"] for r in R})
def n_complete(js):
    have={}
    for r in R:
        if r["judge"] in js and r.get("p") is not None: have.setdefault((r["submission"],r["instance_id"]),set()).add(r["judge"])
    return sum(1 for v in have.values() if len(v)==len(js))
best=["deepseek-v4-flash","glm-5.2","gpt-oss-120b"]
for name,js in [("all10",J),("google3",["gemma-3-27b","gemma-4-e4b","gemma-4-e2b"]),("best3",best),("best3+e4b",best+["gemma-4-e4b"])]:
    e=A.ensemble(R,lab,js); print(name,{k:(round(v,4) if isinstance(v,float) else v) for k,v in e.items()},"n_complete",n_complete(js))
print("glm on best3 items:",)
items={k for k in lab if all(rows.get((k[0],k[1],j),{}).get("p") is not None for j in best)}
g=[(rows[(a,b,"glm-5.2")]["p"],lab[(a,b)]) for a,b in items]
print("GLM-5.2 alone on same",len(items),"items: AUROC",round(A.auroc([p for p,_ in g],[y for _,y in g]),4),"Brier",round(A.brier([p for p,_ in g],[y for _,y in g]),4))
