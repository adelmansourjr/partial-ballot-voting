# An exact certificate from partial approval counts

A voter can answer whether they approve a named proposal, or whether they approve at most a specified number from the fixed list. Each answer is yes or no. Individual answers constrain particular entries. Count answers maintain an interval of feasible ballot sizes. The goal is to recover a fixed tie-broken proportional approval voting (PAV) committee whenever all feasible full profiles yield that same outcome.

## Sharp committee margins with an interval

Fix a proposed committee W, a rival C, and one voter. Partition unresolved entries into proposals shared by W and C, proposals in C alone, proposals outside both, and proposals in W alone. The last three buckets have capacities c, o and w. Let y denote the number of confirmed approvals anywhere in the ballot and let [L,U] be the current interval on the total number of approvals.

Consider an admissible number s of approvals from the shared bucket. The residual count over the three other buckets has endpoints

```text
l = max(0, L - y - s)
u = min(U - y - s, c + o + w).
```

Skip this s if l > u. Any feasible completion puts at most min(c,u) approvals in C alone and at least max(0,l-c-o) in W alone. Both extremes are attained together. Set the residual total to max(l,min(c,u)). Fill the C-only bucket first, then the outside bucket, and finally the W-only bucket. If u < c, the chosen total is u and all approvals go to C alone. If u ≥ c, the method fills C alone and sends the minimum required overflow into W alone. This gives the largest possible score of C and the smallest possible score of W conditional on s, because a PAV contribution increases with the number of approved committee members.

Maximising the difference over s yields the tight single-voter margin. The number of shared counts checked is at most k+1 for a committee of size k. Once bucket capacities and known counts are available, this is O(k) arithmetic per voter and ordered committee pair. It also applies to any nondecreasing score that depends on each voter's number of approved winners. Each voter has an independent completion constraint, so each rival's maximal profile margin is the sum of these single-voter extrema. Add the appropriate tie correction in scaled integer PAV units. The proposed committee is the unique deterministic outcome for all completions exactly when no rival can beat it under that correction. [Proof of the base completion criterion and references](THEORY.md).

The finite benchmark enumerates small feasible ballot sets to evaluate the working posterior for query selection. The O(k) claim concerns the certificate margin once capacities are known. This distinction leaves a clear computational question for larger ballots.

## A response-count separation with equal-size answers

Take n voters, m proposals and k seats, with 2 ≤ k < m. On the witness profile, all voters approve only the first proposal A. PAV with the fixed lexicographic tie rule selects the first k proposals.

A method restricted to individual approval questions must read all n(m−k) outside cells on this profile. An unread outside cell could be flipped to yes while leaving every observed answer unchanged. The resulting profile makes a committee containing A and that outside proposal win. A transcript containing all those answers as no is also sufficient. Every committee containing an outside proposal can replace it with an unused inside proposal without losing score, and the first k proposals precede all such alternatives in ties.

An expanded method can ask each voter “Do you approve at most one proposal?” and “Do you approve A?” The two yes answers settle the entire row. Continuing with individual questions on other profiles gives a universally correct method, and the witness finishes in at most 2n responses. Each question has a binary answer.

For m−k ≥ 2, two questions per row are also necessary on this profile. Suppose one voter has received at most one question. If it concerns individual approval, an outside approval can be added while preserving that answer and every other voter's ballot. With at least two outside proposals, an outside proposal other than the named one is available. If it concerns a count threshold, replacing the voter's sole approval of A with a sole outside approval leaves the answer exactly the same. In both cases an alternative committee containing A and the outside proposal beats the first k proposals. Closing a transcript therefore requires at least two answers from every voter. If m−k = 1, one outside-cell question per voter attains the lower bound of one question per row. The two interface optima on the witness are **n(m−k)** and **n min(m−k, 2)**. For the benchmark sizes this is **36 versus 24 binary responses**. The certificate provides a concrete design advantage when count thresholds are easy to answer. A comparison of human time or effort still requires measuring the cost of each question type.

The audit checks every single outside-cell flip and finds counterexamples after each of the 120 single-question alternatives. It independently enumerates 351 interval ballot states and 1,640 complete profiles across 75 tiny partial problems. [Runnable audit](binary_queries.py).
