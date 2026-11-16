# Can a yes/no question about a total help certify a vote?

The [original study](README.md) compared numeric approval totals with individual approval questions. This follow-up gives both question types **one binary answer**. A procedure may ask whether a voter approves a named proposal or whether they approve at most a specified number. It can certify the proportional approval voting (PAV) committee only when every ballot completion gives that same tie-broken winner.

## The exact result

With twelve voters who all approve just the first proposal, five proposals and two seats, the best possible individual-approval method needs **36 answers**. The best possible mixed method needs **24 answers**. Ask each voter whether they approve at most one proposal and whether they approve the first. Both answers are yes and settle the entire row. For the lower bound, one whole row can still contain an outside approval after any single permitted answer on that row, while keeping every other ballot fixed. The outside approval changes the winning committee. The audit checks all 120 single-question alternatives.

For general transcripts, count questions give a lower and upper bound on each ballot's size. Conditional on the number of unknown approvals shared between two committees, there is an exact worst-case margin. A single allocation reaches the rival's highest feasible approval count and the incumbent's lowest at the same time. This gives a certificate using at most k+1 shared counts per voter and committee pair. [Derivation and exact question-count bounds](BINARY_QUERY_THEORY.md).

## Frozen comparison

We compared the two question sets with the same tie order, posterior, exact certificate and active query rule. The rule estimates how much a question can remove uncertainty or narrow the most threatening rivals' worst-case score differences. Its estimate comes from a uniform distribution over feasible rows. The certificate considers every feasible completion. We fixed the primary selector on 48 development profiles. Its mean approval-only certification cost was **49.92** questions versus **51.67** for expected-margin variance, so the primary comparison uses the stronger development selector in both question sets. [Protocol](BINARY_QUERY_PROTOCOL.md) · [All results and profile-level endpoint records](binary_query_results.json).

On 120 fresh clustered profiles, opening the question set to both kinds took **50.42** answers on average versus **50.08** with individual approval questions. The average saving was **−0.34 answers**, with paired 95% interval **[−0.77, +0.08]**. This missed the five percent threshold. The extra question type was used 7.43 times per profile on average. At the fixed checkpoint of 30 questions, the expected-score incumbent's mean PAV regret fell from 0.46 to 0.33. Neither policy had certified any winner by then. This secondary endpoint scores the interim candidate. Every final certificate agreed with the full-ballot result.

| Panel | Approval questions only | Mixed questions | Answers saved |
|---|---:|---:|---:|
| Clustered, 120 | 50.08 | 50.42 | −0.34 |
| Independent utility, 36 | 51.69 | 52.75 | −1.06 |
| Polarised, 36 | 48.00 | 48.69 | −0.69 |
| Near consensus, 36 | 34.00 | 33.89 | +0.11 |
| Five historical approval sources, equal-source mean | 52.52 | 52.41 | +0.11 |

In the clustered panel, enabling the query type under the alternate expected-variance selector also increased question count, by 0.45 (paired 95% interval 0.05 to 0.91). Charging 1.5 approval-question units per count question increased primary mean cost to **50.28** and left 0.34 such questions per profile. At a price of 2 the primary selector chose only approval questions. All prices measure modelled burden. Each actual count-threshold response is one bit and its effort could vary substantially.

The historical holdout ran 20 fresh random 12-voter, five-proposal projections on each of five pinned Pabulib approval sources. Source-level mean differences ranged from **1.50 more** to **1.10 fewer** questions. The source-level aggregate describes these records and this projection design. Their original elections used their own rules and constraints. [Public ballot source](https://pabulib.org/) · [Input hashes and download instructions](README.md#replay).

The exact saving on the witness and the small measured changes under active querying motivate a sharper research question. On an entirely unobserved ballot, the uniform prior is exchangeable across candidates. Conditioning on a threshold answer alone leaves all size-k committees with the same expected ballot score, so an isolated threshold has zero immediate expected-margin variance reduction. On the witness, combining it with a proposal-specific answer resolves the ballot. Can an exact two-request planning rule recognise that complementarity and preserve its advantage once human effort is measured? For a discussion with Martin Bullinger, the 36-to-24 example is the clean proof to show first, followed by this empirical boundary. [Supervisor notes and advert](SUPERVISOR_NOTES.md).

## Replay

The original project provides the pinned dataset download step in its [replay instructions](README.md#replay). Use that dataset directory below. The audit invokes no downloads and checks 351 small interval states, 1,640 tiny profile completions, and the exact question-count separation. The full replay verifies the saved JSON byte-for-byte.

```sh
python binary_queries.py --audit-only
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python binary_queries.py \
  --verify binary_query_results.json --data-dir "$voting_data"
```
