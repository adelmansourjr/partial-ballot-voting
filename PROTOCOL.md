# Certifying a committee from partial approvals

Local prospective protocol frozen before computing comparative outcomes. The primary fixture is a community panel choosing three of seven equally eligible proposals using proportional approval voting (PAV). A voter approves an arbitrary subset. Each individual response reveals one candidate's approval bit. Answers are truthful and deterministic. With public cardinalities, the panel also knows each voter's exact total number of approvals. A fixed lexicographic tie order is published before elicitation. The full-information reference maximises the sum of H_j, with H_0 = 0 and H_j the j-th harmonic number.

## Question and thresholds

Can an exact worst-case margin over shared ballots certify the same tie-broken winner with fewer questions than separate committee-score intervals, holding both the questions and information fixed? Primary estimand is mean paired difference in yes/no responses, intervals minus coupled, divided by the complete profile's 154 cells. Primary gate is a gain of at least 0.04 and a 95% paired bootstrap lower endpoint above zero on the 96 clustered profiles. Query order is one uniform permutation independent of the votes. Both certificates update after the same answers in that permutation, skipping already known or logically forced cells. Charge one extra response per voter to the count-first modes in a secondary comparison against the count-unknown exact mode. Report separate count and binary burden since a count reveals a different amount of information.

Secondary comparisons ask whether choosing queries improves on the fixed uniform schedule when all methods use the same exact certifier, and whether counts still help accounting for the extra response. Compare certificate-aware query selection with an expected-margin variance baseline. Each adaptive baseline evaluates only legal unresolved cells with both outcomes feasible. Both use the posterior mean-score incumbent and a uniform distribution over each row's feasible completions as a working query model. This posterior has full support and determines the candidate to verify and the next query, while the certificate considers every feasible completion. The design treats the posterior as a query model.

- **Expected-margin variance.** Take the expected-score incumbent's six closest challengers in expected score. Average the reduction in variance of their margins against the incumbent, with weights of 1/(1 + expected gap in PAV units). Compute the reduction by conditional expectation given each possible approval answer, exactly for a voter, with independent feasible ballot rows.
- **Certificate aware.** Take the six challengers with highest coupled worst-case margin plus the tie correction, considering rivals that can defeat the incumbent. Use the expected reduction in this margin from one yes/no response, weighting each by 1/(1 + distance below the most adverse margin in PAV units). For equal priority, break ties using a fixed uniform permutation. Both adaptive methods use the same permitted information, same update and exact certifier and the full catalogue of possible questions.

A complete transcript with a fixed query schedule is examined for the first time each certificate passes. Threshold for a caught contradiction is zero errors for every declared winner under every profile. Report rates certified by 0.25, 0.50 and 0.75 of all cells, counts-to-certificate per profile, paired means, and fixed-budget actual PAV regret of the posterior-mean output just after the 0.5 budget or by exact stopping, whichever comes first. Use 5,000 seeded paired percentile bootstrap resamples at the profile level. Interval calculations condition on the specified generator. Keep the adaptive comparisons and stress cohorts secondary.

## Synthetic profiles and external validation

Generate 22 voters, seven candidates and three seats. Randomise candidate order once per profile and keep it shared across methods. Latent preference blocks use three factions and two specialised proposals per faction plus one shared proposal.

- Clustered, 96 profiles. Faction probabilities (0.49, 0.31, 0.20). Utilities are independent N(0,1) plus +2.3 to own pair and +0.75 to shared proposal. Ballot lengths 1, 2, 3, 4 with probabilities (0.18, 0.38, 0.32, 0.12).
- Uniform, 48. Utilities independent N(0,1), same lengths.
- Polarised, 48. Uniform faction probability with own pair +3.6 and shared item +0.20, same lengths.
- Near consensus, 48. Utilities independent N(0,1), add +2.0 to all three common proposal utilities, same lengths.

Use NumPy's PCG64 and explicit seed bands starting at 640000, 650000, 660000, 670000 respectively. Initial ballots have no revealed yes/no cells. The fixed order and tie permutation for queries use the same seed plus 80000 stream across methods. Draw 12 independent seeded 22-ballot samples per eligible source below, without replacement within draw and randomly selecting exactly seven candidates per draw. Select candidates and ballots uniformly. Project each directly recorded ballot onto those seven candidates and retain projected ballots with zero approvals. Eligibility requires at least seven projects, at least 22 ballots and vote type approval. Sources are listed irrespective of eligibility. Use seeds from 690000, offset by source index times 100. Report per-source descriptive effects and their macro average. Each source is a fixed historical profile. These are hypothetical equal-size decisions replayed over public ballot patterns. The source elections have different costs, feasibility, and rules.

Pinned Pabulib source revision beda04462f2fbee240202510b860106ee9754dd4 of pabulib/pabulib_files

- Canada_Stanford_Dataset_PB_Dieppe_2018_vote_approvals.pb
- France_Toulouse_2019.pb
- Netherlands_Amsterdam_2019_Kleine_Wereld.pb
- Netherlands_Amsterdam_2020_Bloemenbuurt.pb
- Hungary_Budapest_2022_VIII_Jozsefvaros.pb
- Poland_Wieliczka_2023_Green_Budget.pb

## Verification and scope

Verify the exact certificate independently by exhaustively enumerating joint completions of tiny deterministic and seeded profiles and checking all pairwise score maxima. Check the count-aware closed-form margin against explicit ballot completion enumeration for generated row states, with and without counts. Test integer tie direction in both orientations, a strict coupled-versus-separate witness, and the full-reveal reference. Audit truthful no-repeat paid queries, exact final winners and certificate inequalities at the checkpoint. Preserve complete profile-level metrics and deterministic result files. Compare full-information sequential PAV with exact PAV on the same profiles as a diagnostic of changing the decision rule (regret and tie-broken committee disagreements). Exact stopping preserves full-information PAV outcomes under the stated response model, including incentive properties of the underlying choice under complete execution. A hard query cap, refusal, inconsistent count or errors require a different operational guarantee.
