# Exact committee decisions with fewer approval questions

**Start with the [exact separation](BINARY_QUERY_RESULTS.md) and [two-question follow-up](LOOKAHEAD_RESULTS.md).** Run the replays with Python 3.12 and the [pinned NumPy dependency](requirements.txt).

A paper-linked voting project for Martin Bullinger's **Algorithms for Collective Decision Making: Voting and Coalition Formation**. A community panel chooses three of seven proposals. We compare the cost of certifying the exact proportional approval voting (PAV) committee when voters' approval bits are available one at a time. This separates the voting rule from the process used to gather its input. [Advert and research notes](SUPERVISOR_NOTES.md).

**Approval totals changed the cost more than a tighter mathematical bound.** On 96 clustered profiles, asking each voter for their approval total reduced mean response burden from **148.93 to 128.36**, counting all 22 numeric answers. The paired reduction was **20.56 answers**, with a 95% interval of **[19.22, 21.92]**. This was a prespecified secondary comparison. The primary comparison found only **1.03** binary answers saved by exact margins over separate score intervals on the same transcript. Its 95% interval was **[0.68, 1.41]** and its practical threshold was **6.16**. Every certified committee matched the full-ballot result. [Frozen protocol](PROTOCOL.md) · [Complete outcomes](results.json).

**Equal-bandwidth follow-up.** A question about whether a voter approves at most a specified number also has just one yes/no answer. We derived a tight interval-count certificate and proved that the mixed interface reduces the exact optimum from 36 to 24 questions on a concrete profile. The measured savings did not carry across clustered and historical ballot projections under our matched active selectors. [Follow-up and replay](BINARY_QUERY_RESULTS.md).

**Two-question planning.** Optimising the second question conditional on the first reply still yielded little average benefit from threshold questions on 120 fresh clustered profiles. The matched two-step mixed interface saved −0.34 answers on average, with a 95% interval of −1.08 to +0.41. Every policy run certified the correct committee. [Matched policies and robustness checks](LOOKAHEAD_RESULTS.md).

## What the certificate proves

PAV gives a voter who approves j elected proposals a harmonic score of 1 + 1/2 + … + 1/j. Compare the expected-score incumbent with every rival using the largest rival-minus-incumbent score that can occur under any completion of the partial profile. The same approval answer affects both scores. Our calculation keeps that dependency and can also enforce a fixed number of approvals per voter. Split a voter's unknown candidate approvals into shared, rival-only, incumbent-only and outside buckets. For each feasible shared approval count, fill the other buckets in the order that favours the rival most. Summing the row maxima yields an exact profile bound. A fixed tie order is built into the integer comparison. [Proof and assumptions](THEORY.md).

There is a sharp illustration of the value of knowing approval totals. For 22 voters who each approve only the same proposal, an exact method that sees only approval bits must read at least **88** answers on this profile. A count-first transcript can certify using **44 responses**, including every numeric answer. Any unread outside bit in the count-free case has a possible completion that makes some different committee win. The demo checks all 88 counterfactuals and the early stop. [Runnable example](demo.py).

## Matched comparisons

Mean number of paid binary answers needed to certify the exact winner is shown below. Each count-first arm also collects exactly 22 numeric answers.

| Profile family | Separate intervals, fixed order | Exact margins, same fixed order | Exact margins without counts | Expected-margin variance, exact stop | Certificate-aware queries, exact stop |
|---|---:|---:|---:|---:|---:|
| Clustered, 96 | 107.40 | 106.36 | 148.93 | **98.31** | 98.79 |
| Uniform, 48 | 107.40 | 106.25 | 149.81 | 100.10 | **99.06** |
| Polarised, 48 | 108.81 | 107.85 | 150.19 | 98.77 | **97.63** |
| Near consensus, 48 | 97.33 | 90.77 | 119.52 | **68.10** | 80.77 |
| Public ballot projections, 5 source mean | 66.40 | 65.58 | 149.43 | 60.78 | **60.48** |

The two adaptive selectors use the same information and consider every unresolved cell. The first measures how a question would reduce uncertainty about expected margins. The second measures how it would reduce the upper bound of the current leading threats. On the clustered profiles the certificate-aware method saved **7.57** answers versus fixed order, with a paired interval **[6.02, 9.13]**. Its advantage over expected-margin variance was **−0.48 [−1.86, 0.90]**. At the half-budget checkpoint average PAV regret was **0.55** for certificate-aware and **0.28** for variance. On near-consensus profiles variance saved **12.67** questions versus certificate-aware, **[9.69, 15.44]**. On the same profiles the primary certificate ablation saved **6.56** questions on average. These secondary effects depend on profile structure and the question model.

The count-first benefit is conditional on response cost. On clustered profiles the fixed exact policy saved 42.56 binary answers before charging the numeric responses. Its mean break-even price was therefore **1.93 binary responses per numeric response**. The experiment treats this as a cost comparison in a simulator. A human study would need to measure effort, errors and willingness to disclose exact totals.

Full-information sequential PAV was a separate decision-rule diagnostic. It disagreed with tie-broken exact PAV on **7 of 96** clustered profiles, with mean PAV score loss **0.0243**. This explains why a smaller query budget and a sequential rule need separate tests. On each eligible real-data projection we used a fixed historical approval source, sampled 22 ballots and seven projects, and masked their recorded answers. Across the five source means, charging every count still saved **61.85 total responses** against count-free fixed exact elicitation. Projecting to seven projects makes some ballots sparse. This is a descriptive stress test of hypothetical equal-size choices. The actual budget allocations use other constraints and rules. [Public source and attribution](https://pabulib.org/) · [Pinned inputs](https://github.com/pabulib/pabulib_files/tree/beda04462f2fbee240202510b860106ee9754dd4/pb_files).

## Replay

The reference run used Python 3.12.8 and NumPy 2.3.5. The demo and exact proof audit use generated inputs. For the full replay, obtain the six pinned Pabulib files. Five contain eligible approval ballots. The Toulouse file is cumulative and records the protocol's eligibility check. The program verifies the raw input fingerprints, uses only project IDs and approval cells, and saves aggregate outputs.

```sh
python -m pip install numpy==2.3.5
python demo.py
python voting.py --audit-only

voting_data="$(mktemp -d)"
python - "$voting_data" <<'PY'
from pathlib import Path
import subprocess
import sys
from voting import REVISION, SOURCES

data_directory = Path(sys.argv[1])
base_url = f"https://raw.githubusercontent.com/pabulib/pabulib_files/{REVISION}/pb_files"
for filename, checksum in SOURCES:
    subprocess.run(
        ["curl", "-q", "--fail", "--silent", "--show-error",
         f"{base_url}/{filename}", "-o", str(data_directory / filename)],
        check=True,
    )
PY
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python voting.py --data-dir "$voting_data" --verify results.json
```

An independent enumerator checked **4,040** complete profiles across 96 small partial profiles and tested each of their 288 row states against the formula. The deterministic full replay covers 240 generated profiles and 60 resamples of public ballots. A natural next question for Bullinger is whether the value of this approval-count interface survives nonresponse or unequal query costs across voters. Both changes require revisiting the exact completion assumptions. [Supervisor reading notes](SUPERVISOR_NOTES.md).
