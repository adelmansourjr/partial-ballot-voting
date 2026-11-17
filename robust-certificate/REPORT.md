# When can one changed approval alter a PAV election?

In **75 of 128 fresh clustered elections (58.6%)**, changing one reported approval could change the proportional approval voting (PAV) winner. Only **42 of 128 (32.8%)** allowed the same change to strictly benefit the voter whose ballot was altered. The distinction held across **100 projections from five historical ballot sources**. At least one change affected the outcome in **50** projections and strictly benefited its voter in **17**. These are possible counterfactuals under specified rules and utilities. The historical records provide no evidence of actual manipulation. [Full-disclosure protocol](FULL_DISCLOSURE_PROTOCOL.md) · [Per-profile results](full_disclosure_results.json)

## Exact certificates with incomplete reports

Consider an incomplete transcript of approval bits. Unreported bits can take either value. A clean profile may disagree with up to $b$ reported bits. For a proposed winning committee $C$ and rival $D$, let $g_i(c)$ be the largest difference between voter $i$'s PAV scores for $D$ and $C$ among ballots that disagree with exactly $c$ reports from that voter. The sharp largest total difference is

$$
M_b(C,D) = \max_{c_i \ge 0,\,\sum_i c_i \le b} \sum_i g_i(c_i).
$$

Every feasible profile gives a feasible choice of costs in this expression. Conversely, a separate voter ballot can attain each term, and assembling those ballots meets the shared budget. These two inclusions establish equality. A short budget convolution computes the maximum and reconstructs the ballots that attain it. Certify $C$ when $M_b(C,D) < 0$ for all earlier rivals and $M_b(C,D) \le 0$ for all later rivals. The inequalities track the lexicographic tie rule. Failure produces a feasible profile where $C$ loses. The budget stage takes $O(nb^2)$ operations per candidate pair after constructing row tables. This is an exact result for enumerated committees and small candidate counts.

The strongest comparator allows each voter to mix row options subject to the budget in expectation. Its dual is $\min_{\lambda\ge 0} \lambda b + \sum_i \max_c (g_i(c)-\lambda c)$. This convex, piecewise-linear function can be minimised exactly over its breakpoints, and the answer can be rounded down in scaled integer units without losing validity. The simpler bound adds one PAV point per potentially incorrect report to each zero-error margin. Related work studies participation incentives, incomplete approval ballots or robustness to vote perturbations. Our model specifies incomplete binary reports with a shared error budget and explicitly distinguishes an outcome change from a profitable unilateral change. [Participation incentives](https://ojs.aaai.org/index.php/AAAI/article/view/28810) · [Incomplete approval voting](https://ojs.aaai.org/index.php/AAAI/article/download/20440/20199) · [Approval voting robustness](https://link.springer.com/article/10.1007/s10058-025-00385-1)

### Three-row separation for one rival

Take the complete approval profile ABCDE, ABCDF, ABCEF, compare target ABC to rival DEF, and allow two altered reports. In integer units of 1/6 of a PAV point, each row's gap table at costs zero through three is (−2, 0, 3, 9). Spending two bits on one row gives a gain of five. Spending one on each of two rows gives four. The sharp gap is therefore −1 unit, or −1/6 point. Enumerating all 172 admissible profiles gives the same maximum.

The fractional relaxation can spend two expected bits by taking the cost-three option on one row with probability 2/3. This gives a gap of 4/3 units, or 2/9 point. All other row slopes from zero are at most 11/3, so that relaxed gap is also sharp. Other committees still compete with ABC in the fixture. The separation concerns the pairwise margin.

## Empirical test

In the first experiment, all methods received 194 of 216 approval bits. At the primary error budget of two and 160 clustered profiles, **all three returned zero certificates**. At zero errors the exact method certified two and both certificates failed with two errors. Exact and fractional coverage were equal throughout all synthetic and historical panels. The practical 3-point threshold was missed. The paired bootstrap interval of 0 to 0 points describes the observed all-abstain panel, with limited information about performance at higher disclosure. [90% protocol](PROTOCOL.md) · [Results](results.json)

The follow-up used fresh seeded draws at 98% and 100% disclosure. For each full profile we evaluated all 216 possible one-bit changes and counted a strict gain when the changed committee contained more candidates sincerely approved by the voter whose report changed. We separately required both winners to be unique, to check whether gains relied entirely on the tie order. Across **63,072** brute-force deviations, the exact certificate classified one-bit vulnerability correctly in all 292 profiles.

| Full disclosure panel | Profiles | One bit changes the winner | Voter can strictly gain | Strict gain with unique winners |
|---|---:|---:|---:|---:|
| Clustered synthetic | 128 | 75 | 42 | 22 |
| Independent synthetic | 64 | 48 | 30 | 20 |
| Dieppe | 20 | 10 | 7 | 4 |
| Amsterdam Kleine Wereld | 20 | 8 | 2 | 2 |
| Amsterdam Bloemenbuurt | 20 | 8 | 3 | 1 |
| Budapest Józsefváros | 20 | 8 | 5 | 1 |
| Wieliczka Green Budget | 20 | 16 | 0 | 0 |

The primary clustered vulnerability estimate was **58.6%**, with a **50.0% to 67.2%** profile-bootstrap interval. A 20-bit uniform random audit would miss an expected **16.9%** of the vulnerable clustered profiles and **6.4%** for the equal-source mean on the historical draws, conditional on vulnerability. Ninety-eight percent disclosure reduced one-bit certificate coverage in clustered profiles from **53 of 128** to **27 of 128**. Exact, fractional, and global Lipschitz coverage still tied across the evaluated panels. The value of the exact method here is the realised counterexample, the exact margin and the ability to distinguish manipulation from sensitivity. No stronger empirical coverage claim follows.

The public panels were projected into a six-candidate, three-seat format and reinterpreted with PAV. Overlapping projections from one archive are not independent elections. The strategic check assumes full information, approval-count utility and a single altered bit. The study does not establish what voters knew or did. A concrete next experiment would use the vulnerable ballots to select which reports to verify, and test against margin and random selection using the same verification budget.

## Reproduce

Use Python with NumPy 2.3.5 and the pinned [Pabulib archives](../README.md#replay). Run from the repository root.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python robust-certificate/robust.py \
  --data-dir "$voting_data" --verify robust-certificate/results.json
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python robust-certificate/full_disclosure.py \
  --data-dir "$voting_data" --verify robust-certificate/full_disclosure_results.json
```

The first script independently enumerates 1,728 exact margin checks and verifies returned witnesses. The follow-up additionally checks the pairwise separation and exhaustively cross-checks the one-bit winner calls.
