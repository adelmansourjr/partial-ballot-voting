"""Certify deterministic PAV decisions from only the approval cells needed."""
import argparse
from dataclasses import dataclass
from functools import cache
import csv
from fractions import Fraction
import hashlib
from io import StringIO
from itertools import combinations, product
import json
from math import floor, lcm
from pathlib import Path
from urllib.request import urlopen

import numpy as np


REVISION = "beda04462f2fbee240202510b860106ee9754dd4"
PROTOCOL_HASH = "b88f23d37201f25a764c7d12a1c636c3c6c18d061da024339bfb289163e7f8a6"
SOURCES = (
    ("Canada_Stanford_Dataset_PB_Dieppe_2018_vote_approvals.pb", "1e7a7423bf7609afc21adff5536dbb8614a5514e79ea895829b79a2ad80e59ed"),
    ("France_Toulouse_2019.pb", "da4b9aeb770c941712cf25051aa619960d7534be205d2c8b83b7f47dd0a8c9cc"),
    ("Netherlands_Amsterdam_2019_Kleine_Wereld.pb", "68e1b541ee2659a02a95d7bc5abd80c28d89a0e043eb3cc9fa3f7e669d1a488c"),
    ("Netherlands_Amsterdam_2020_Bloemenbuurt.pb", "ed22796aab0e6586ae3c5446b5cdb1658217e91fd42c04c91f2a6fff72950f34"),
    ("Hungary_Budapest_2022_VIII_Jozsefvaros.pb", "14e8548f94cb251942ac82198ed80378e8a20a556160ecfcf01136aab51f3c3f"),
    ("Poland_Wieliczka_2023_Green_Budget.pb", "f2d9c4485b3f9205f513231ae1ebd05c05abd37917e6473942d762069f2d1221"),
)
VOTERS = 22
CANDIDATES = 7
SEATS = 3
BANDS = {"clustered": (640000, 96), "uniform": (650000, 48), "polarised": (660000, 48), "near_consensus": (670000, 48)}
RUNNERS = ("separate_intervals", "coupled_fixed", "coupled_without_counts", "expected_variance", "certificate_aware")


@dataclass
class Row:
    averages: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    challenger_margins: np.ndarray
    approval_probabilities: np.ndarray


class SearchSpace:
    def __init__(self, n_candidates, seats):
        self.n_candidates = n_candidates
        self.seats = seats
        self.all = (1 << n_candidates) - 1
        self.ballots = np.arange(1 << n_candidates)
        self.cells = 1 << np.arange(n_candidates)
        self.pop = np.array([i.bit_count() for i in self.ballots.tolist()])
        self.committees = tuple(combinations(range(n_candidates), seats))
        self.masks = np.array([sum(1 << c for c in indices) for indices in self.committees])
        self.scale = lcm(*range(1, seats + 1))
        self.values = np.cumsum([0] + [self.scale // j for j in range(1, seats + 1)])
        self.score = self.values[self.pop[self.ballots[:, None] & self.masks]]
        self.shared = self.masks[:, None] & self.masks
        self.first_only = self.masks[:, None] & ~self.masks
        self.second_only = self.masks & ~self.masks[:, None]
        self.outside = self.all ^ (self.masks[:, None] | self.masks)
        self.tie_cost = (
            np.arange(len(self.masks))[None, :] < np.arange(len(self.masks))[:, None]
        ).astype(int)

    def possible(self, yes, no, total):
        feasible = ((self.ballots & yes) == yes) & ((self.ballots & no) == 0)
        if total >= 0:
            feasible &= self.pop == total
        return self.ballots[feasible]

    def max_margins(self, yes, no, total):
        """Largest challenger minus target score for every committee pair."""
        unresolved = self.all ^ (yes | no)
        yes_counts = self.pop[self.masks & yes]
        shared_capacity = self.pop[self.shared & unresolved]
        first_capacity = self.pop[self.first_only & unresolved]
        second_capacity = self.pop[self.second_only & unresolved]
        outside_capacity = self.pop[self.outside & unresolved]
        maxima = np.full(self.shared.shape, -100, dtype=int)
        for shared in range(self.seats + 1):
            feasible = shared <= shared_capacity
            if total >= 0:
                remaining = total - yes.bit_count() - shared
                feasible &= (remaining >= 0) & (
                    remaining <= first_capacity + second_capacity + outside_capacity
                )
                second = np.minimum(second_capacity, remaining)
                first = np.maximum(0, remaining - second_capacity - outside_capacity)
            else:
                second = second_capacity
                first = 0
            target_count = np.clip(yes_counts[:, None] + shared + first, 0, self.seats)
            challenger_count = np.clip(yes_counts + shared + second, 0, self.seats)
            gap = self.values[challenger_count] - self.values[target_count]
            maxima = np.maximum(maxima, np.where(feasible, gap, -100))
        if np.any(maxima == -100):
            raise ValueError("Inconsistent ballot observations")
        return maxima

    @cache
    def row(self, state):
        feasible = self.possible(*state)
        if not len(feasible):
            raise ValueError("Inconsistent ballot observations")
        scores = self.score[feasible]
        probabilities = ((feasible[:, None] & self.cells) != 0).mean(axis=0)
        return Row(scores.mean(axis=0), scores.min(axis=0), scores.max(axis=0),
                   self.max_margins(*state), probabilities)

    def score_leader(self, expected):
        # Keep tie-breaking stable under roundoff in sums of small exact integers.
        return int(np.flatnonzero(expected >= expected.max() - 1e-9)[0])

    def best_sequential(self, ballots):
        current = np.zeros(len(ballots), dtype=int)
        selected = []
        candidates = list(range(self.n_candidates))
        while len(selected) < self.seats:
            gains = {}
            for candidate in candidates:
                positive = (ballots & (1 << candidate)) != 0
                gains[candidate] = int(
                    (self.values[current + positive] - self.values[current]).sum()
                )
            best = min(candidates, key=lambda candidate: (-gains[candidate], candidate))
            candidates.remove(best)
            selected.append(best)
            current += (ballots & (1 << best)) != 0
        return self.committees.index(tuple(sorted(selected)))


def contacted(state, candidate, answer):
    yes, no, total = state
    mask = 1 << candidate
    assert not (yes | no) & mask
    return (yes | mask, no, total) if answer else (yes, no | mask, total)


class Elicitation:
    def __init__(self, space, ballots, counts):
        self.space = space
        self.states = [(0, 0, int(ballot).bit_count() if counts else -1) for ballot in ballots]
        self.rows = [space.row(state) for state in self.states]
        self.expected = np.sum([row.averages for row in self.rows], axis=0)
        self.lower = np.sum([row.minimum for row in self.rows], axis=0)
        self.upper = np.sum([row.maximum for row in self.rows], axis=0)
        self.margins = np.sum([row.challenger_margins for row in self.rows], axis=0)
        self.ballots = ballots
        self.count = 0

    def candidate(self):
        return self.space.score_leader(self.expected)

    def interval_certificate(self, leader):
        margins = self.upper - self.lower[leader] + self.space.tie_cost[leader]
        margins[leader] = 0
        return not int(margins.max())

    def exact_certificate(self, leader):
        margins = self.margins[leader] + self.space.tie_cost[leader]
        return not int(margins.max())

    def query(self, voter, candidate):
        before = self.rows[voter]
        assert 0.0 < before.approval_probabilities[candidate] < 1.0
        observed = bool(self.ballots[voter] & (1 << candidate))
        state = contacted(self.states[voter], candidate, observed)
        after = self.space.row(state)
        self.expected += after.averages - before.averages
        self.lower += after.minimum - before.minimum
        self.upper += after.maximum - before.maximum
        delta = after.challenger_margins - before.challenger_margins
        assert np.all(delta <= 0)
        self.margins += delta
        self.rows[voter] = after
        self.states[voter] = state
        self.count += 1

    def unresolved(self, voter, candidate):
        return 0.0 < self.rows[voter].approval_probabilities[candidate] < 1.0

    def select(self, mode, leader, order):
        space = self.space
        if mode == "certificate_aware":
            threat = self.margins[leader] + space.tie_cost[leader]
            rivals = [i for i in np.argsort(-threat, kind="stable") if i != leader and threat[i] > 0][:6]
            maximum = threat[rivals[0]]
            weights = 1 / (1 + (maximum - threat[rivals]) / space.scale)
        else:
            rivals = [i for i in np.argsort(-self.expected, kind="stable") if i != leader][:6]
            weights = 1 / (1 + np.maximum(0, self.expected[leader] - self.expected[rivals]) / space.scale)
        best, best_value = None, -1.0
        for voter, candidate in order:
            if not self.unresolved(voter, candidate):
                continue
            row = self.rows[voter]
            state = self.states[voter]
            p = row.approval_probabilities[candidate]
            yes = space.row(contacted(state, candidate, True))
            no = space.row(contacted(state, candidate, False))
            if mode == "certificate_aware":
                prior = row.challenger_margins[leader, rivals]
                after = p * yes.challenger_margins[leader, rivals] + (1 - p) * no.challenger_margins[leader, rivals]
                information = prior - after
            else:
                change = (yes.averages[rivals] - yes.averages[leader]) - (no.averages[rivals] - no.averages[leader])
                information = p * (1 - p) * change**2
            value = float(weights @ information)
            if value > best_value + 1e-10:
                best, best_value = (voter, candidate), value
        if best is None:
            raise RuntimeError("Exact search failed to certify a fully revealed profile")
        return best


def fixed_run(space, ballots, order, counts, reference):
    state = Elicitation(space, ballots, counts)
    stops = {}
    budget = len(order) // 2
    for position in range(len(order) + 1):
        leader = state.candidate()
        if "exact" not in stops and state.exact_certificate(leader):
            assert leader == reference
            stops["exact"] = state.count
        if counts and "intervals" not in stops and state.interval_certificate(leader):
            assert leader == reference and "exact" in stops
            stops["intervals"] = state.count
        if "budget_leader" not in stops and (state.count == budget or "exact" in stops):
            stops["budget_leader"] = leader
        if "exact" in stops and (not counts or "intervals" in stops):
            return stops
        voter, candidate = order[position]
        if state.unresolved(voter, candidate):
            state.query(voter, candidate)
    raise RuntimeError("Fixed schedule did not find the full-information result")


def adaptive_run(space, ballots, order, mode, reference):
    state = Elicitation(space, ballots, True)
    budget = len(order) // 2
    budget_leader = None
    while state.count <= len(order):
        leader = state.candidate()
        certified = state.exact_certificate(leader)
        if budget_leader is None and (state.count == budget or certified):
            budget_leader = leader
        if certified:
            assert leader == reference
            return state.count, budget_leader
        voter, candidate = state.select(mode, leader, order)
        state.query(voter, candidate)
    raise RuntimeError("Adaptive schedule did not terminate")


def order_for(seed):
    order = list(product(range(VOTERS), range(CANDIDATES)))
    return [order[index] for index in np.random.default_rng(seed + 80000).permutation(len(order))]


def evaluate(space, ballots, seed):
    full_scores = space.score[ballots].sum(axis=0)
    reference = int(full_scores.argmax())
    sequential = space.best_sequential(ballots)
    order = order_for(seed)
    fixed = fixed_run(space, ballots, order, True, reference)
    unknown = fixed_run(space, ballots, order, False, reference)
    variance = adaptive_run(space, ballots, order, "expected_variance", reference)
    aware = adaptive_run(space, ballots, order, "certificate_aware", reference)
    steps = (fixed["intervals"], fixed["exact"], unknown["exact"], variance[0], aware[0])
    # Report separate intervals at its paired decision under the exact policy.
    # Both checks share a transcript and posterior-mean output at the fixed cap.
    leaders = (fixed["budget_leader"], fixed["budget_leader"],
               unknown["budget_leader"], variance[1], aware[1])
    regret = [(int(full_scores[reference] - full_scores[leader])) / space.scale for leader in leaders]
    return {
        "seed": seed,
        "queries": dict(zip(RUNNERS, steps)),
        "regret_at_half_budget": dict(zip(RUNNERS, regret)),
        "sequential_pav_disagrees": sequential != reference,
        "sequential_pav_regret": int(full_scores[reference] - full_scores[sequential]) / space.scale,
    }


def generated_profile(seed, family):
    rng = np.random.default_rng(seed)
    utility = rng.normal(size=(VOTERS, CANDIDATES))
    if family in ("clustered", "polarised"):
        probabilities = (0.49, 0.31, 0.20) if family == "clustered" else (1 / 3,) * 3
        strength, shared = (2.3, 0.75) if family == "clustered" else (3.6, 0.20)
        factions = rng.choice(3, size=VOTERS, p=probabilities)
        utility[np.arange(VOTERS), 2 * factions] += strength
        utility[np.arange(VOTERS), 2 * factions + 1] += strength
        utility[:, -1] += shared
    elif family == "near_consensus":
        utility[:, :3] += 2.0
    lengths = rng.choice((1, 2, 3, 4), size=VOTERS, p=(0.18, 0.38, 0.32, 0.12))
    relabel = rng.permutation(CANDIDATES)
    return np.array([
        sum(1 << int(relabel[candidate]) for candidate in np.argsort(values)[-length:])
        for values, length in zip(utility, lengths)
    ], dtype=int)


def sections(archive):
    output = {}
    current = None
    for line in csv.reader(StringIO(archive.decode("utf-8-sig")), delimiter=";"):
        if not line or not any(line):
            continue
        row = [value.strip() for value in line]
        if row[0] in ("META", "PROJECTS", "VOTES") and len(row) == 1:
            current = row[0]
            output[current] = []
        else:
            output[current].append(row)
    return output


def split_voting_data(archive):
    parsed = sections(archive)
    metadata = dict(parsed["META"][1:])
    projects = parsed["PROJECTS"]
    votes = parsed["VOTES"]
    n_projects, n_votes = int(metadata["num_projects"]), int(metadata["num_votes"])
    if n_projects != len(projects) - 1 or n_votes != len(votes) - 1:
        raise ValueError("Unexpected Pabulib table sizes")
    if metadata["vote_type"] != "approval" or n_projects < CANDIDATES or n_votes < VOTERS:
        return metadata, (), ()
    project_ids = tuple(row[projects[0].index("project_id")] for row in projects[1:])
    vote_column = votes[0].index("vote")
    approvals = tuple(frozenset(row[vote_column].split(",")) - {""} for row in votes[1:])
    if not all(set(project_ids).issuperset(vote) for vote in approvals):
        raise ValueError("Vote contains an unknown project")
    return metadata, project_ids, approvals


def external_profile(seed, projects, approvals):
    rng = np.random.default_rng(seed)
    candidates = [projects[index] for index in rng.choice(len(projects), CANDIDATES, replace=False)]
    voters = rng.choice(len(approvals), VOTERS, replace=False)
    return np.array([
        sum(1 << c for c, project in enumerate(candidates) if project in approvals[index])
        for index in voters
    ], dtype=int)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def download_sources(path):
    path.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCES:
        destination = path / name
        if destination.exists():
            continue
        url = f"https://raw.githubusercontent.com/pabulib/pabulib_files/{REVISION}/pb_files/{name}"
        with urlopen(url, timeout=45) as reply:
            data = reply.read()
        if digest(data) != expected:
            raise ValueError(f"Unexpected SHA-256 for {name}")
        destination.write_bytes(data)


def audit():
    space = SearchSpace(4, 2)
    rng = np.random.default_rng(901227)
    scores = np.array([
        [int(space.scale * sum((Fraction(1, j) for j in range(1, sum(bool(ballot & (1 << c)) for c in committee) + 1)), Fraction()))
         for committee in space.committees]
        for ballot in range(16)
    ])
    np.testing.assert_array_equal(scores, space.score)
    largest = 0
    completion_count = 0
    for _ in range(96):
        row_states = []
        for _ in range(3):
            true_ballot = int(rng.integers(16))
            revealed = int(rng.integers(16))
            yes, no = true_ballot & revealed, (15 ^ true_ballot) & revealed
            count = true_ballot.bit_count() if rng.integers(2) else -1
            state = yes, no, count
            row_states.append(state)
        feasible_banks = []
        total = np.zeros((len(space.masks), len(space.masks)), dtype=int)
        for yes, no, count in row_states:
            possible = [b for b in range(16) if b & yes == yes and not b & no and
                        (count == -1 or b.bit_count() == count)]
            feasible_banks.append(possible)
            analytic = space.max_margins(yes, no, count)
            brute = np.max([scores[b][None, :] - scores[b][:, None] for b in possible], axis=0)
            np.testing.assert_array_equal(analytic, brute)
            total += analytic
        completions = list(product(*feasible_banks))
        largest = max(largest, len(completions))
        completion_count += len(completions)
        completion_scores = [sum((scores[b] for b in ballots), np.zeros(len(space.masks), dtype=int)) for ballots in completions]
        full_matrix = np.max([s[None, :] - s[:, None] for s in completion_scores], axis=0)
        np.testing.assert_array_equal(full_matrix, total)
        certified = np.flatnonzero((total + space.tie_cost).max(axis=1) == 0).tolist()
        winners = {int(s.argmax()) for s in completion_scores}
        assert certified == (list(winners) if len(winners) == 1 else [])
    # Shared candidate 0, approved candidate 2 and rejected candidates 1 and 3.
    # (0,2) is always the deterministic winner despite the earlier rival (0,1)
    # having an upper score equal to its lower score.
    witness = space.row((0b0100, 0b1010, -1))
    lead = space.committees.index((0, 2))
    rival = space.committees.index((0, 1))
    assert witness.challenger_margins[lead, rival] == -1
    assert witness.maximum[rival] - witness.minimum[lead] == 0
    assert max(witness.challenger_margins[lead] + space.tie_cost[lead]) == 0
    assert max((witness.maximum - witness.minimum[lead] + space.tie_cost[lead])[i]
               for i in range(len(space.masks)) if i != lead) > 0
    assert (space.tie_cost[lead, rival], space.tie_cost[rival, lead]) == (1, 0)
    return {"tiny_profiles": 96, "single_ballot_states": 288, "joint_completions": completion_count,
            "largest_tiny_profile": largest, "strict_witness": {"target": [0, 2], "rival": [0, 1],
                                                                 "challenger_margin_pav_units": -0.5,
                                                                 "separate_margin_pav_units": 0}}


def summarise(records, rng):
    by_runner = {name: np.array([row["queries"][name] for row in records]) for name in RUNNERS}
    cells = VOTERS * CANDIDATES
    pairs = {
        "intervals_minus_coupled_on_same_answers": by_runner["separate_intervals"] - by_runner["coupled_fixed"],
        "fixed_minus_certificate_aware": by_runner["coupled_fixed"] - by_runner["certificate_aware"],
        "variance_minus_certificate_aware": by_runner["expected_variance"] - by_runner["certificate_aware"],
        "without_counts_minus_counts_including_numeric_answers": by_runner["coupled_without_counts"] - by_runner["coupled_fixed"] - VOTERS,
    }
    resamples = rng.integers(0, len(records), size=(5000, len(records)))
    paired = {}
    for name, differences in pairs.items():
        intervals = np.quantile(differences[resamples].mean(axis=1), (0.025, 0.975))
        paired[name] = {"responses_saved": float(differences.mean()),
                        "interval_responses": intervals.tolist(), "fraction_of_cells_saved": float(differences.mean() / cells),
                        "interval_fraction_of_cells": (intervals / cells).tolist()}
    summary = {}
    for name in RUNNERS:
        values = by_runner[name]
        numeric = 0 if name == "coupled_without_counts" else VOTERS
        summary[name] = {
            "mean_yes_no_responses": float(values.mean()),
            "median_yes_no_responses": float(np.median(values)),
            "mean_total_responses": float(values.mean() + numeric),
            "mean_regret_at_half_binary_budget": float(np.mean([row["regret_at_half_budget"][name] for row in records])),
            "certificate_rate_at_binary_budget": {
                str(fraction): float((values <= floor(cells * fraction)).mean())
                for fraction in (0.25, 0.50, 0.75)
            },
        }
    summary["full_information_sequential_pav"] = {
        "committee_disagreement": float(np.mean([row["sequential_pav_disagrees"] for row in records])),
        "pav_regret": float(np.mean([row["sequential_pav_regret"] for row in records])),
    }
    return {"size": len(records), "mean_metrics": summary, "paired_effects": paired}


def run(data_dir):
    assert digest(Path(__file__).with_name("PROTOCOL.md").read_bytes()) == PROTOCOL_HASH
    audit_result = audit()
    space = SearchSpace(CANDIDATES, SEATS)
    generator = np.random.default_rng(191874)
    results = {"protocol_sha256": PROTOCOL_HASH, "source_revision": REVISION,
               "proof_audit": audit_result, "synthetic": {}, "public_ballot_stress": {}}
    for family, (first, size) in BANDS.items():
        records = [evaluate(space, generated_profile(seed, family), seed) for seed in range(first, first + size)]
        summary = summarise(records, generator)
        results["synthetic"][family] = {"summary": summary, "profiles": records}
        print(f"{family} | {size} complete", flush=True)
    primary = results["synthetic"]["clustered"]["summary"]["paired_effects"]["intervals_minus_coupled_on_same_answers"]
    results["primary_gate"] = primary["fraction_of_cells_saved"] >= 0.04 and primary["interval_responses"][0] > 0
    source_means = {name: [] for name in RUNNERS}
    for i, (name, expected) in enumerate(SOURCES):
        if data_dir is None:
            continue
        archive = (data_dir / name).read_bytes()
        if digest(archive) != expected:
            raise ValueError(f"Unexpected SHA-256 for {name}")
        metadata, projects, approvals = split_voting_data(archive)
        entry = {"sha256": expected, "vote_type": metadata["vote_type"],
                 "number_of_projects": int(metadata["num_projects"]), "number_of_ballots": int(metadata["num_votes"])}
        if projects:
            first = 690000 + 100 * i
            records = [evaluate(space, external_profile(seed, projects, approvals), seed)
                       for seed in range(first, first + 12)]
            # Source resamples are descriptive: no intervals from recycled ballots.
            summary = summarise(records, generator)
            for method in RUNNERS:
                source_means[method].append(summary["mean_metrics"][method]["mean_yes_no_responses"])
            descriptive = {key: value["responses_saved"] for key, value in summary.pop("paired_effects").items()}
            entry.update({"eligible": True, "summary": summary,
                          "mean_paired_response_savings": descriptive, "samples": records})
        else:
            entry["eligible"] = False
        results["public_ballot_stress"][name] = entry
        print(f"{name} | parsed and checked", flush=True)
    if data_dir is not None:
        results["public_macro_mean_yes_no_responses"] = {name: float(np.mean(values))
                                                         for name, values in source_means.items()}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Optional path to the pinned public source tables")
    parser.add_argument("--download", action="store_true", help="Fetch public source tables into --data-dir")
    parser.add_argument("--audit-only", action="store_true")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=Path)
    output.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit(), sort_keys=True))
        return
    if args.download:
        if args.data_dir is None:
            parser.error("--download requires --data-dir")
        download_sources(args.data_dir)
    if args.output is None and args.verify is None:
        parser.error("give --output or --verify")
    results = run(args.data_dir)
    payload = json.dumps(results, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        assert payload == args.verify.read_text(), "Replay differs from expected results"
        print("Verified complete matching voting results.", flush=True)


if __name__ == "__main__":
    main()
