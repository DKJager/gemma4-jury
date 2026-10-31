# A Jury of Makers: Can Independent Open Models, Including Gemma 4 on a CPU, Tell Whether a Coding Agent's Patch Works?

*Judging SWE-bench Verified patches from the issue and the diff alone, with ten open models from seven makers*

**Alan Finney, 3Dogs Nexus**

## Abstract

Coding agents produce patches faster than anyone can review or test them. We ask whether open models can act as a cheap first filter: given only an issue's text and an agent's patch, with no repository and no tests, can they predict whether the patch resolves the issue? We sampled 300 patches from four public agent submissions to SWE-bench Verified (150 that resolved their issue and 150 that did not) and asked ten open models from seven makers for a probability of resolution. The best single judge, GLM-5.2, reached an AUROC of 0.838 (95% CI 0.789–0.880). Diversity of makers mattered more than size of the panel. Across all 98 possible three-judge panels drawn from three different makers, the median AUROC was 0.759, and 93.9% of those panels beat a same-maker panel of three Google models (0.691). The best cross-maker panel matched the best single judge on discrimination (0.840 against 0.838 on the same items) and was better calibrated (Brier 0.207 against 0.225). Gemma 4 E4B, run quantized on four CPU cores with no GPU, scored 0.669, well above the server-hosted Gemma 3 27B (0.557) and level with Mistral-Small-3.1 (0.666). Used as a pre-test gate, a unanimous vote of the best panel accepted 56.3% of patches at 70.4% precision, against a 50% base rate. All code, prompts, samples and judgments are released at github.com/DKJager/gemma4-jury.

## 1. Introduction

Agentic software engineering has moved the bottleneck from writing patches to checking them. A coding agent can propose many candidate fixes for one issue; running a full test suite for each is slow, and for many real repositories a suite that exposes the bug does not exist yet. A reviewer that reads the issue and the diff and says "this probably works" or "this probably does not" would be valuable even if it is imperfect, as long as its confidence means something.

Large language models are already used as judges of model output. Two questions matter for using them on code. First, which judges, and how many? A panel of judges can cancel out individual errors only if those errors are not shared. Models from the same maker share training data, recipes and failure modes, so we expect a panel from different makers to be more useful than a panel of siblings. Second, can this run where developers are? The Gemma 4 Developer Agent competition asks about agents on consumer hardware. If a small model running on a laptop-class CPU contributes a useful vote, a local pre-test gate becomes practical.

We contribute:

1. A measured comparison of ten open judges from seven makers on a balanced SWE-bench Verified sample, with bootstrap confidence intervals and calibration.
2. Evidence that **maker diversity** drives jury quality: across every possible three-maker panel, 93.9% beat the same-maker Google panel.
3. A CPU-only result for **Gemma 4 E4B and E2B**: E4B, quantized to 4 bits on four CPU cores, outperformed a server-hosted Gemma 3 27B by 0.11 AUROC.
4. A **pre-test gate** recipe with its precision and coverage, and released artifacts for reproduction.

## 2. Related work

**SWE-bench and SWE-bench Verified.** SWE-bench (Jimenez et al., 2023) evaluates whether a system can resolve real GitHub issues, judged by hidden tests. SWE-bench Verified is a human-validated subset of 500 issues released by OpenAI in 2024. Public submissions publish their patches and per-instance results, which gives us ground-truth labels for real agent patches.

**Patch correctness prediction.** Predicting whether a generated patch is correct without running it has a history in automated program repair. Tian et al. (2020) evaluated learned representations of code changes for this task. These approaches are trained on labelled patches; ours is zero-shot and uses general-purpose models.

**LLMs as judges and juries.** Zheng et al. (2023) studied strong models as judges of open-ended answers. Verga et al. (2024) proposed replacing a single large judge with a panel of smaller models from different families ("juries"), reporting better agreement with humans and less intra-model bias. We test the same idea on code patches, where ground truth comes from test execution rather than human preference, and we isolate maker diversity directly by comparing every cross-maker panel with a same-maker panel.

## 3. Method

### 3.1 Data

We loaded the per-instance patches (`all_preds.jsonl`) and results (`results/results.json`) of four public agent submissions to SWE-bench Verified from the official `SWE-bench/experiments` repository. The four submissions are `20241202_agentless-1.5_claude-3.5-sonnet-20241022` (81 sampled items), `20240728_sweagent_gpt4o` (81), `20251120_livesweagent_gemini-3-pro-preview` (70) and `20250415_openhands` (68). Empty patches were dropped. From the pool we drew a balanced sample of 300 (submission, instance) pairs, 150 labelled resolved and 150 unresolved, spreading picks round-robin across the 12 repositories within each label so that no repository dominates (seed 2026). The label is the official test-based outcome from each submission's results file.

### 3.2 Judging protocol

Each judge received a system prompt and a user message containing the issue's `problem_statement` and the agent's `model_patch`, each truncated to 6,000 characters with a visible marker. Judges were blind to which agent wrote the patch and saw no tests, repository files or execution results. They were asked to return JSON of the form `{"p_resolved": <0..1>, "reason": "..."}` at temperature 0. Hosted judges could write up to 900 output tokens; for the three reasoning models (DeepSeek-V4-Flash, GLM-5.2, gpt-oss-120b) we raised this to 8,000 so their reasoning did not cut off the final JSON. We parsed the first JSON object containing `p_resolved` and clamped it to [0, 1]. Replies without one are counted as **missing**, not imputed; coverage is reported for every judge.

### 3.3 Judges

| Judge | Maker | Where it ran |
|---|---|---|
| GLM-5.2 | Zhipu | Nebius Token Factory |
| gpt-oss-120b | OpenAI (open-weight) | Nebius Token Factory |
| DeepSeek-V4-Flash | DeepSeek | Nebius Token Factory |
| Qwen3-235B-A22B-Instruct-2507 | Alibaba | Nebius Token Factory |
| Gemma 3 27B-it | Google | Nebius Token Factory |
| Mistral-Small-3.1-24B-Instruct | Mistral | IBM watsonx.ai |
| Llama-4-Maverick-17B-128E-Instruct (FP8) | Meta (open-weight) | IBM watsonx.ai |
| Granite-4-H-Small | IBM | IBM watsonx.ai |
| **Gemma 4 E4B-it** | Google | **CPU only**, Q4_0 QAT GGUF, llama.cpp |
| **Gemma 4 E2B-it** | Google | **CPU only**, Q4_0 QAT GGUF, llama.cpp |

**Gemma 4 on consumer-class hardware.** The two Gemma 4 judges ran in a llama.cpp server container capped at 4 CPUs and 6 GB of RAM (3 threads, context 4,096 tokens, thinking disabled, at most 400 output tokens), on a general-purpose cloud VM with no GPU. E4B processed prompts at about 48 tokens per second and generated at about 7.6 tokens per second, roughly 84 seconds per patch. Only the edge variants were judged; the 26B and 31B variants did not fit the memory budget. The models are distributed under Apache-2.0, and the licence notices are kept with the artifacts.

### 3.4 Metrics

- **AUROC** of `p_resolved` against the label (Mann–Whitney; ties count one half), with a 95% bootstrap interval from 500 resamples of each judge's scored items.
- **Brier score** of `p_resolved` (lower is better).
- **Coverage**: the share of the 300 items with a parsed probability.
- **Ensembles**: the mean `p_resolved` of the listed judges, computed only on items where every listed judge returned a probability. The number of items differs between ensembles, and we report it.
- **Agreement curve**: accept a patch when at least *k* judges give `p_resolved ≥ 0.5`; report coverage (the share of items accepted) and precision (the share of accepted items that truly resolved). We fixed the vote threshold at 0.5, the natural "more likely than not" cut, before looking at results, and did not tune it.

## 4. Results

### 4.1 Single judges

| Judge | AUROC [95% CI] | Brier | Items scored (coverage) |
|---|---|---|---|
| GLM-5.2 | **0.838** [0.789, 0.880] | **0.227** | 268 (89.3%) |
| gpt-oss-120b | 0.775 [0.722, 0.827] | 0.249 | 300 (100%) |
| DeepSeek-V4-Flash | 0.746 [0.700, 0.793] | 0.269 | 294 (98.0%) |
| Qwen3-235B | 0.705 [0.646, 0.755] | 0.335 | 300 (100%) |
| **Gemma 4 E4B (CPU)** | 0.669 [0.604, 0.725] | 0.335 | 284 (94.7%) |
| Mistral-Small-3.1 | 0.666 [0.602, 0.727] | 0.335 | 299 (99.7%) |
| Llama-4-Maverick | 0.626 [0.582, 0.669] | 0.398 | 299 (99.7%) |
| **Gemma 4 E2B (CPU)** | 0.577 [0.536, 0.619] | 0.400 | 256 (85.3%) |
| Granite-4-H-Small | 0.577 [0.524, 0.624] | 0.367 | 300 (100%) |
| Gemma 3 27B | 0.557 [0.519, 0.595] | 0.447 | 297 (99.0%) |

Every judge beats chance (0.5), but the spread is wide: from 0.838 down to 0.557. Gemma 4 E4B on a CPU sits in the middle of the field. Its interval lies entirely above Gemma 3 27B's, so the gap of 0.11 AUROC is not a sampling artefact. It is level with Mistral-Small-3.1 (0.669 against 0.666, overlapping intervals) and ahead of Llama-4-Maverick and Granite-4-H-Small. E2B is weaker, and its 44 unparseable replies give it the lowest coverage.

### 4.2 Juries: maker diversity matters

| Jury (mean p) | AUROC | Brier | Items with every judge |
|---|---|---|---|
| All 10 judges | 0.766 | 0.275 | 217 |
| Same-maker: Gemma 3 27B + Gemma 4 E4B + Gemma 4 E2B | 0.691 | 0.349 | 247 |
| Best cross-maker: DeepSeek-V4-Flash + GLM-5.2 + gpt-oss-120b | **0.840** | **0.207** | 263 |
| Best cross-maker + Gemma 4 E4B | 0.802 | 0.225 | 248 |
| GLM-5.2 alone, on the same 263 items as the best triple | 0.838 | 0.225 | 263 |

There are 98 ways to choose three judges from three different makers. Their median AUROC is 0.759 (range 0.656–0.840), and **93.9% of them beat the same-maker Google panel**. The Google panel is better than any of its members alone, but it stays well below the typical cross-maker panel.

Averaging all ten judges is not the best choice either. The weaker judges pull the mean down (0.766), and requiring a probability from all ten leaves only 217 items. A small panel of strong judges from different makers works better.

The best cross-maker panel ties the best single judge on discrimination (0.840 against 0.838 on identical items) and is **better calibrated** (Brier 0.207 against 0.225). Its probabilities can be taken more nearly at face value, which matters for a gate.

Adding Gemma 4 E4B to that panel lowered both AUROC (to 0.802) and calibration (Brier 0.225). With equal weights, a fourth judge that is weaker than the other three dilutes the panel. This argues for choosing jurors by measured skill and maker diversity, not simply adding more.

### 4.3 Agreement as a pre-test gate

| Rule (accept if ≥ k judges say p ≥ 0.5) | Coverage | Precision |
|---|---|---|
| Best cross-maker triple, k = 1 | 82.0% | 60.2% |
| Best cross-maker triple, k = 2 | 75.7% | 62.6% |
| Best cross-maker triple, **k = 3 (unanimous)** | **56.3%** | **70.4%** |
| All 10 judges, k = 5 | 91.3% | 54.4% |
| All 10 judges, k = 8 | 76.0% | 62.7% |
| All 10 judges, k = 10 (unanimous) | 43.3% | 70.8% |

The base rate is 50%. A unanimous vote of three strong judges from different makers accepts more than half the patches and raises precision by about 20 points. Unanimity across all ten judges reaches the same precision (70.8%) but accepts far fewer patches (43.3%). The smaller, diverse panel gives the better trade-off.

## 5. Discussion

**Why maker diversity works.** A mean only helps when the judges' errors differ. Three Google models share a lineage and, it appears, many of the same blind spots: the same-maker panel improves on its members but stays far below the typical cross-maker panel. We do not attribute the gap to any architectural feature. We only observe that panels mixing makers were better in 92 of 98 cases.

**What Gemma 4 on a CPU offers.** A 4-bit Gemma 4 E4B on four CPU cores, without a GPU, scored better than a server-hosted Gemma 3 27B and level with a hosted 24B Mistral model. At about 84 seconds per patch it is too slow to judge thousands of patches interactively, but fast enough to screen an agent's few candidate fixes for one issue while tests are being prepared, entirely offline. In our data it did not improve the strongest hosted panel. Its value is as a local juror when hosted models are unavailable or the code cannot leave the machine.

**How a coding agent could use this.**
1. Generate candidate patches.
2. Ask a small cross-maker jury for `p_resolved` on each candidate.
3. Send unanimous accepts to test execution first, and send split decisions to a human or a deeper check.

On our sample, step 3 makes 70.4% of the first tests run on patches that truly resolve the issue, compared with 50% without a gate.

**Generality.** The protocol needs only an issue, a diff and a label, so it applies to any repository with history. The finding about maker diversity is about the judges, not the task, and should be tested on other judgment problems.

## 6. Limitations

- **Sample size and source.** 300 balanced items from four agent submissions; the 50% base rate is by construction, while real agent pass rates differ, which changes precision at any threshold.
- **Truncated, static view.** Judges saw at most 6,000 characters of the issue and of the patch, no repository context and no tests. Multi-file fixes and long issues are disadvantaged.
- **Missing outputs.** Unparseable replies are treated as missing: GLM-5.2 covered 89.3% of items and Gemma 4 E2B 85.3%. If missing replies are not random, single-judge scores are biased. Ensembles use only items every member scored, so their item counts differ (217–263).
- **Gemma 4 scope.** Only the E4B and E2B edge variants, 4-bit, CPU-only. An earlier run lost rows when the container was killed for exceeding the 6 GB cap at a context of 8,192 tokens (the key-value cache grows with context). We reran at a context of 4,096, which still fits the truncated prompt (about 2,700 tokens) and the 400-token answer, with automatic container restart. The final counts are those reported. Attempts to use a free hosted Gemma 4 tier failed with rate-limit errors and were excluded entirely.
- **One prompt, temperature 0, equal weights.** We did not tune prompts per judge or learn ensemble weights; both could change the ranking.
- **Bootstrap intervals** are per judge; we did not correct for multiple comparisons across the 98 panels.

## 7. Reproducibility

The package `g4paper` contains `data.py` (loads submissions and labels), `sample.py` (balanced round-robin sampling), `judge.py` (prompt, parser, resumable judging) and `analysis.py` (AUROC, Brier, ensembles, agreement curve), with 8 passing unit tests. The artifacts are the sample (`sample_ids.json`), every raw judgment with the judge's reason (`run/judgments_*.jsonl`), the analysis output (`run/results_analysis.json`) and the model licence notices (`run/licenses/`). Seed 2026 throughout.

## 8. Conclusion

Open models can serve as a useful, imperfect first reader of coding-agent patches without running anything. Who sits on the jury matters more than how many: panels drawn from different makers beat a same-maker panel in 94% of cases, and the best three-maker panel matched the best single judge while being better calibrated. Gemma 4 E4B, quantized on four CPU cores, outperformed a hosted Gemma 3 27B, so a credible local juror is within reach of ordinary hardware. A unanimous three-maker gate raised the share of truly working patches among those sent to testing from 50% to 70%.

## References

- Jimenez, C. E., et al. (2023). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* arXiv:2310.06770.
- OpenAI (2024). *Introducing SWE-bench Verified.*
- Tian, H., et al. (2020). *Evaluating Representation Learning of Code Changes for Predicting Patch Correctness in Program Repair.* ASE 2020.
- Verga, P., et al. (2024). *Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models.* arXiv:2404.18796.
- Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* arXiv:2306.05685.
- Google (2026). Gemma 4 model cards and QAT GGUF releases (Hugging Face `google/gemma-4-E4B-it-qat-q4_0-gguf`, `google/gemma-4-E2B-it-qat-q4_0-gguf`), Apache-2.0.
- SWE-bench/experiments repository, github.com/SWE-bench/experiments.
