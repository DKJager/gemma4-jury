"""Judge runner. usage: python run_judges.py sample|nebius|ibm|g4local [limit]
Env: NEBIUS_API_KEY, WATSONX_API_KEY, WATSONX_PROJECT_ID; g4local needs G4_JUDGE and G4_PORT (a local llama.cpp server)."""
import sys, os, json, time, urllib.request, urllib.error, http.client
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(ROOT, "src"))
from g4paper import data, sample, judge
D = os.path.join(ROOT, "data"); RUN = os.path.join(ROOT, "run"); os.makedirs(RUN, exist_ok=True)
SUBS = sorted(os.listdir(f"{D}/subs")) if os.path.isdir(f"{D}/subs") else []
NEB = {"gpt-oss-120b": "openai/gpt-oss-120b", "qwen3-235b": "Qwen/Qwen3-235B-A22B-Instruct-2507",
       "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash-0731", "glm-5.2": "zai-org/GLM-5.2", "gemma-3-27b": "google/gemma-3-27b-it"}
IBMJ = {"ibm-granite-4-small": "ibm/granite-4-h-small", "ibm-llama4-maverick": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8", "ibm-mistral-small": "mistralai/mistral-small-3-1-24b-instruct-2503"}
IBM_LANE_CAP = int(os.environ.get("WATSONX_TOKEN_CAP", "3300000"))  # stop before a token budget is exhausted
def make_sample():
    issues = data.load_issues(f"{D}/verified_issues.jsonl"); items = []
    for s in SUBS:
        for iid, v in data.load_submission(f"{D}/subs/{s}").items():
            if iid in issues:
                items.append({"instance_id": iid, "submission": s, "repo": issues[iid]["repo"], "resolved": bool(v["resolved"])})
    pick = sample.balanced_sample(items, 300, 2026)
    pm = {s: data.load_submission(f"{D}/subs/{s}") for s in SUBS}
    for it in pick:
        it["patch"] = pm[it["submission"]][it["instance_id"]]["patch"]; it["problem_statement"] = issues[it["instance_id"]]["problem_statement"]
    json.dump(pick, open(f"{RUN}/sample.json", "w"))
    from collections import Counter
    print("pool", len(items), "sample", len(pick), Counter(p["resolved"] for p in pick), Counter(p["submission"] for p in pick))
def neb_key():
    return os.environ["NEBIUS_API_KEY"]
def neb_fn(model, key):
    def ask(messages):
        body = {"model": model, "messages": messages, "max_tokens": 8000 if any(x in model for x in ("DeepSeek","GLM","gpt-oss")) else 900, "temperature": 0}
        r = urllib.request.Request("https://api.tokenfactory.nebius.com/v1/chat/completions", data=json.dumps(body).encode(),
                                   headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(r, timeout=180)); m = d["choices"][0]["message"]
        return m.get("content") or m.get("reasoning_content") or ""
    return ask
_IBM = {"tok": None, "exp": 0.0, "key": None}
LEDGER = f"{RUN}/ibm_ledger.jsonl"
def ibm_used():
    try: return sum(int(json.loads(l).get("tokens") or 0) for l in open(LEDGER))
    except FileNotFoundError: return 0
def ibm_token():
    import urllib.parse
    if _IBM["tok"] and _IBM["exp"] - time.time() > 300: return _IBM["tok"]
    if not _IBM["key"]:
        _IBM["key"] = os.environ["WATSONX_API_KEY"].strip()
    data = urllib.parse.urlencode({"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": _IBM["key"]}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("https://iam.cloud.ibm.com/identity/token", data=data, headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}), timeout=60))
    _IBM["tok"] = r["access_token"]; _IBM["exp"] = float(r.get("expiration") or time.time() + 3000); return _IBM["tok"]
def ibm_fn(model):
    def ask(messages):
        if ibm_used() >= IBM_LANE_CAP: raise RuntimeError("ibm lane token cap reached")
        body = {"model_id": model, "project_id": os.environ["WATSONX_PROJECT_ID"], "messages": messages, "max_tokens": 900, "temperature": 0}
        t0 = time.time()
        r = urllib.request.Request("https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2024-10-10", data=json.dumps(body).encode(),
                                   headers={"Authorization": "Bearer " + ibm_token(), "Content-Type": "application/json", "Accept": "application/json"})
        d = json.load(urllib.request.urlopen(r, timeout=300)); u = d.get("usage") or {}
        with open(LEDGER, "a") as f: f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": model, "tokens": u.get("total_tokens") or 0, "secs": round(time.time() - t0, 1), "lane": "judges"}) + "\n")
        return (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return ask
if __name__ == "__main__":
    mode = sys.argv[1]; lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    STOP_AFTER = 5   # stop a judge after 5 consecutive errors instead of burning a rate limit
    def guard(fns):
        streak = {j: 0 for j in fns}
        def wrap(j, f):
            def ask(m):
                if streak[j] >= STOP_AFTER: raise RuntimeError("judge stopped after %d consecutive errors" % STOP_AFTER)
                try:
                    out = f(m); streak[j] = 0; return out
                except Exception:
                    streak[j] += 1; raise
            return ask
        return {j: wrap(j, f) for j, f in fns.items()}
    if mode == "sample": make_sample(); sys.exit()
    items = json.load(open(f"{RUN}/sample.json"))[:lim]
    if mode == "nebius":
        k = neb_key(); fns = {j: neb_fn(m, k) for j, m in NEB.items()}
        judge.run_judgments(items, list(NEB), guard(fns), f"{RUN}/judgments_nebius.jsonl", {j: 0.5 for j in NEB})
    elif mode == "ibm":
        fns = {j: ibm_fn(m) for j, m in IBMJ.items()}
        judge.run_judgments(items, list(IBMJ), guard(fns), f"{RUN}/judgments_ibm.jsonl", {j: 0.5 for j in IBMJ})
    elif mode == "g4local":   # Gemma 4 edge variants served by a local llama.cpp server on 127.0.0.1
        j = os.environ["G4_JUDGE"]; port = os.environ["G4_PORT"]
        def ask(messages):
            body = {"messages": messages, "max_tokens": 400, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                       headers={"Content-Type": "application/json"})
            for attempt in range(4):   # container restarts (e.g. out-of-memory) are transient: wait and retry
                try:
                    m = json.load(urllib.request.urlopen(r, timeout=900))["choices"][0]["message"]
                    return m.get("content") or m.get("reasoning_content") or ""
                except (ConnectionError, urllib.error.URLError, http.client.HTTPException) as e:
                    if attempt == 3: raise
                    time.sleep(45)
                    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                               headers={"Content-Type": "application/json"})
        judge.run_judgments(items, [j], guard({j: ask}), f"{RUN}/judgments_gemma4.jsonl", {j: 0.0})
    print("done", mode)
