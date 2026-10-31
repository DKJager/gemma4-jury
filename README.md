# gemma4-jury

Code, data and results for the paper **"A Jury of Makers: Can Independent Open Models, Including Gemma 4 on a CPU, Tell Whether a Coding Agent's Patch Works?"**, written for the Kaggle *Gemma 4 Developer Agent* paper track. The paper is in `paper/draft_v2.md`.

**Question.** Given only an issue's text and a coding agent's patch, with no repository and no tests, can open models predict whether the patch resolves the issue? And does a panel of judges from *different makers* do better than a panel from one maker?

**Answer, in brief** (300 SWE-bench Verified patches, 150 resolved / 150 not):

| | AUROC |
|---|---|
| Best single judge (GLM-5.2) | 0.838 |
| Best 3-maker panel (DeepSeek-V4-Flash + GLM-5.2 + gpt-oss-120b) | 0.840 (Brier 0.207 vs 0.225) |
| Median of all 98 three-maker panels | 0.759 |
| Same-maker panel (Gemma 3 27B + Gemma 4 E4B + Gemma 4 E2B) | 0.691 |
| Gemma 4 E4B, Q4_0, **4 CPU cores, no GPU** | 0.669 |
| Gemma 3 27B, hosted | 0.557 |

A unanimous vote of the best 3-maker panel accepts 56.3% of patches at 70.4% precision (base rate 50%).

## Reproduce the analysis (no API keys needed)

```bash
python -m pytest tests -q          # 8 passed
python run/runan.py                # writes run/results_analysis.json
python run/ens_n.py                # ensemble item counts, like-for-like comparison
```
Python 3.12, standard library only.

## Files

| Path | What |
|---|---|
| `src/g4paper/` | `data.py` (load submissions/labels), `sample.py` (balanced sampling), `judge.py` (prompt, parser, resumable runner), `analysis.py` (AUROC, Brier, ensembles, agreement curve) |
| `sample_ids.json` | The 300 sampled (submission, instance_id, repo, resolved) items, seed 2026 |
| `run/judgments_*.jsonl` | Every raw judgment: `p` (probability the patch resolves the issue), the judge's reason, errors. Unparseable replies keep `p = null` |
| `run/results_analysis.json` | All reported metrics |
| `run/licenses/` | Gemma 4 model cards and licence log (Apache-2.0) |
| `run_judges.py` | Runs the judges (`sample`, `nebius`, `ibm`, `g4local`) |
| `g4local.sh` | Serves Gemma 4 E4B/E2B on CPU with llama.cpp in Docker (4 CPUs, 6 GB, context 4096) and judges |

## Re-run the judges

1. Put the four submission folders from [SWE-bench/experiments](https://github.com/SWE-bench/experiments) (`evaluation/verified/<name>/`, with `all_preds.jsonl` and `results/results.json`) in `data/subs/`, and SWE-bench Verified rows as JSONL in `data/verified_issues.jsonl`. Then run `python run_judges.py sample` to rebuild `run/sample.json`. It should match `sample_ids.json`.
2. Hosted judges: set `NEBIUS_API_KEY` (Nebius Token Factory), or `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` (IBM watsonx.ai), then run `python run_judges.py nebius` or `python run_judges.py ibm`.
3. Gemma 4 on CPU: run `./g4local.sh` (needs Docker; downloads the official Q4_0 QAT GGUF weights).

Runs are resumable, and completed (submission, instance, judge) rows are skipped.

## Notes

- Judges see the issue and the patch, each truncated to 6,000 characters, at temperature 0. Output limits: 900 tokens, raised to 8,000 for reasoning models (DeepSeek, GLM, gpt-oss) so their final JSON is not cut off; 400 tokens with thinking disabled for Gemma 4 on CPU.
- Ensembles average `p` over items where every listed judge returned one. Item counts are reported.
- Limitations are listed in section 6 of the paper.

## License

Code: Apache-2.0 (see `LICENSE`). SWE-bench data and agent patches belong to their authors; this repository stores only instance IDs and labels, plus the judges' outputs.

Alan Finney, 3Dogs Nexus (https://3dogs.ai)
