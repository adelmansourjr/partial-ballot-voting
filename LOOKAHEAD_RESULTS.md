# Two-question planning for exact committee certification

On one twelve-voter profile, adding yes/no questions about a voter's total approvals cut the [exact certification minimum](BINARY_QUERY_RESULTS.md) from 36 answers to 24. A myopic heuristic captured little of that gain on other profiles. We tested whether exact planning two questions ahead made the extra question type useful in practice. On 120 fresh clustered synthetic profiles, the matched two-step comparison saved **−0.34 answers** on average, with a paired 95% interval of **[−1.08, +0.41]**. The interval falls below the prespecified practical gain of 2.58 answers, or 5% of the individual-question baseline.

## What the experiment controls

Cross the question menu and the planning horizon, giving four policies. Both menus can ask whether someone approves a particular proposal. The mixed menu also permits a yes/no threshold question about the total. All policies share the exact PAV rule, tie order, uniform working posterior on each feasible row, and prespecified tie order for potential questions. Each check of the output committee considers every feasible profile completion, and the hidden full ballots supply evaluation answers only.

At each real query, freeze the prospective winner, challengers and weights from the original pressure heuristic. The objective is the weighted sum of worst-case score margins for those challengers. The one-step heuristic maximises the expected decrease in that sum. The two-step policy scores the first question and optimises the second separately for each possible answer. After contacting voter i, only that voter's feasible set changes. Consequently, the best second query is the maximum of the remaining options for voter i and the best one-step query on all other rows. This finds the optimal two-step plan for the frozen surrogate without approximating the second query menu. The policy executes the first question, checks the certificate, and replans from the received answer.

## Results

All **1,312 policy runs** over **328 profiles** certified the full-information, tie-broken PAV winner. Values in the table are mean questions. A positive saving favours the mixed interface at a matched horizon.

| Panel | One-step individual | One-step mixed | Two-step individual | Two-step mixed | Two-step answers saved |
|---|---:|---:|---:|---:|---:|
| Clustered synthetic, 120 | 50.23 | 50.68 | 51.69 | 52.03 | −0.34 |
| Independent synthetic, 36 | 52.39 | 52.97 | 54.08 | 53.78 | +0.31 |
| Polarised synthetic, 36 | 48.53 | 49.44 | 51.25 | 51.89 | −0.64 |
| Near-consensus synthetic, 36 | 33.25 | 33.08 | 34.69 | 34.33 | +0.36 |
| Historical projections, 100 across five sources | 53.32 | 52.84 | 54.19 | 52.69 | +1.50 |

In the primary panel, looking ahead used **1.46 more** questions under the individual interface and **1.35 more** under the mixed interface. The interaction was **+0.11** answers saved, with an exploratory paired 95% interval of **[−0.63, +0.82]**. No primary-panel policy certified a winner within 30 answers.

The historical panel used 20 fresh projections per source and gave each source equal weight. Savings across those sources ranged from **−0.30** to **+3.05** answers with two-step planning. At 1.5 units per threshold reply, the mix cost more in all five sources. Repricing the original unit-price transcripts gave a **−4.61-unit** mean saving in the primary panel at that price. These comparisons hold the recorded sequences of questions fixed. The costs model alternative effort assumptions. The projections use recorded public ballots and supply no evidence about actual response times, effort, or missing answers.

The exhaustive two-step optimiser still works with a short-horizon score surrogate. An improvement in the expected two-step margin can be allocated to a challenger whose margin no longer blocks certification later. A policy built around the probability or cost of certifying, rather than the frozen weighted margin, would answer a different question. These results show where the original exact separation remains compelling and delimit what this particular planner can recover.

## Audit and replay

An independent enumerator checked **372** first-action values against feasible leaf outcomes on small partial profiles. The one-step policies matched the original implementation's observed query counts and types on both interfaces across eight development profiles. Every experimental answer preserves the true ballot and weakly tightens the worst-case margins. The protocol and sample bands were frozen before running the final panel. [Frozen protocol](LOOKAHEAD_PROTOCOL.md) · [Per-profile outcomes](lookahead_results.json).

The reference run used Python 3.12 and NumPy 2.3.5. Obtain the pinned [Pabulib files](README.md#replay) for the full replay.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python lookahead.py --audit-only
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python lookahead.py \
  --data-dir "$voting_data" --verify lookahead_results.json
```
