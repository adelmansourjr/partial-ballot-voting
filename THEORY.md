# Exact worst-case margins for partial ballots

A panel publishes the eligible committees and a fixed lexicographic tie order. Let S(W) be the total PAV score of committee W, multiplied by the least common multiple of integers from 1 to the committee size k. This makes each score an integer. Each ballot is consistent with its revealed approval bits and, if supplied, its total number of approvals. Every complete profile is a Cartesian product of feasible voter ballots.

## A tight comparison

For a proposed committee W and a challenger C, let M_v(W,C) be the largest possible single-voter difference S_v(C) − S_v(W) over voter v's feasible ballots. The largest achievable profile difference is exactly the sum of the M_v(W,C). The sum is an upper bound by termwise maximisation. For a fixed challenger, the independent voter constraints let us combine the individual maximisers into a feasible joint profile, which gives equality. The same argument applies to any sum of monotone voter-specific cardinality scores. Imber, Israel, Brill and Kimelfeld use the decomposition for the necessary-committee problem. [Original analysis, §6.1](https://doi.org/10.1016/j.artint.2025.104381).

To compute an individual maximum with exact cardinality, split the unknown cells into four buckets.

| Bucket | Meaning |
|---|---|
| Shared | Candidate in both W and C |
| Challenger only | Candidate in C alone |
| Outside | Candidate in neither |
| Proposed only | Candidate in W alone |

Condition on an integer s, the number of approved unknown cells in the shared bucket. It ranges from zero to at most k. Let r be the outstanding approval count after reserving s for the shared bucket. Reject this value of s if r is negative or exceeds the total capacity of the three other buckets. Otherwise place up to r approvals in the challenger-only bucket, then place the remaining approvals outside both committees, then place any final remainder in the proposed-only bucket. This simultaneously maximises the challenger's approval count and minimises the proposal's, conditional on s, since the same r must fit across the three buckets. Monotonicity of the score makes that allocation optimal for this value of s. Take the largest score difference across feasible s. If the total is unknown, fill the challenger-only bucket and leave the proposed-only one empty while enumerating s. The score objective omits the outside bucket.

The number of shared counts tested is at most k + 1. With bit sets and per-row preprocessing, this yields a simple scan over voters and challenger committees when verifying a proposed decision. The research implementation caches all committee-pair margins for encountered ballot states so its two query policies can compare alternative next questions.

## A complete stopping rule

Add one integer score unit to each challenger's worst-case difference if it precedes W in the declared tie order. W is the full-information answer for every feasible completion exactly when every corrected difference is at most zero. If a corrected difference is positive, the maximising completion for that challenger witnesses a loss for W under the deterministic rule.

For any transcript, the exact rule stops at the earliest point at which some procedure can certify this fixed full-information decision from the same constraints. If the rule cannot certify an outcome, at least two feasible completions yield different deterministic winners. To find a candidate to test, use the expected PAV maximiser under any distribution with full support over the feasible profiles. If W wins deterministically everywhere, it weakly outscores every rival everywhere. Every rival with equality in expectation must tie everywhere, so W also precedes it in the tie order. Therefore looking only at the expected maximiser loses no certificate. The experiment uses uniform feasible ballots solely to select the candidate and choose questions.

The worst-case margin always improves or ties separate score intervals. For any feasible ballot, the challenger's actual score is at most its standalone maximum and the proposed candidate's score is at least its standalone minimum. Those separate extrema may come from different ballot completions. In the product model the exact margin is the best bound for that information. The theorem makes no claim of an optimal query order.

## A strict example

There is one voter, two seats and four candidates ordered A, B, C, D. C is approved, B and D are rejected, and A is unresolved. Candidates A and C are selected in both completions after applying the fixed tie order. The earlier challenger A and B has standalone maximum score 1. The proposed committee A and C has standalone minimum score 1. Separate intervals cannot rule out a decisive tie.

| Unresolved answer on A | Score of A and C | Score of A and B | Challenger minus proposal |
|---|---:|---:|---:|
| No | 1 | 0 | −1 |
| Yes | 1.5 | 1 | −0.5 |

The exact maximum for this rival is −0.5. The coupled rule verifies all challengers and certifies A and C immediately. The independent audit reproduces the arithmetic and tie conditions. [Runner](voting.py).

## An exact query separation

Take n voters, m candidates and k seats with 2 ≤ k < m. On a particular complete profile, every voter approves only A, the first candidate. The lexicographic PAV result is the first k candidates. Suppose a count-free certifier has left one outside-candidate bit unread. Changing just this missing bit to yes creates a feasible completion with a different winner. A committee containing A and that outside candidate gains half a PAV point from one voter. Committees excluding the outside candidate gain zero. An exact certifier must therefore read all n(m−k) outside bits on this profile. Asking those outside bits first reaches the bound. Once those bits are all zero, replacing an inside candidate with an outside candidate can only remove approvals, and the inside committee wins ties.

With an exact approval count of one from each voter, ask every voter whether they approve A. The yes answers logically settle every other bit. This transcript uses 2n responses, including all n count answers. For seven proposals and three seats this is 44 responses for 22 voters versus a count-free lower bound of 88 binary responses. The separation concerns this explicit family and prices each numeric and binary answer at one response. It clarifies why the information interface can matter even when the exact certifier is already optimal for a given transcript. [Runnable construction](demo.py).

## Participation when elicitation completes

If questions and stopping complete with a valid transcript, the mechanism outputs the same lexicographic PAV maximiser for every active electorate as the full-ballot rule. Any preference property defined only through those outcomes therefore transfers. For PAV specifically, if a group gains weakly in approved winners by abstaining and at least one gains strictly, its sum of harmonic contributions is strictly higher on the abstention outcome. Compare the PAV optimality inequalities for the two electorates to get a contradiction. More generally, the published paper establishes group participation for approval-based committee scoring rules under its stated set-preference extension. Our argument uses the fixed tie order and strict increase of harmonic scores. Question costs, refusal, and an imposed deadline can change incentives. [Bullinger et al., Theorem 1 and appendix](https://doi.org/10.1609/aaai.v38i9.28810).
