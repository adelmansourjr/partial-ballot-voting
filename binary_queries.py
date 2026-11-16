"""Exact PAV completion certificates with two types of yes/no questions."""

import argparse
from dataclasses import dataclass
from functools import cache
from itertools import product
import json
from pathlib import Path

import numpy as np

from voting import SOURCES, SearchSpace, digest, split_voting_data


PROTOCOL_SHA256 = "41a58379640bf0531bed59bdf7b993325f56b980398dae7c3b73498771d1b968"
VOTERS = 12
PROPOSALS = 5
SEATS = 2
SELECTORS = ("variance", "pressure")
PANELS = {
    "clustered": (422200, 120),
    "independent": (423200, 36),
    "polarised": (424200, 36),
    "near_consensus": (425200, 36),
}
CHECKPOINT = 60  # Twice the nominal price of 30 single-approval questions.


@dataclass(frozen=True)
class Row:
    completions: np.ndarray
    averages: np.ndarray
    margins: np.ndarray


@dataclass(frozen=True)
class Options:
    ids: np.ndarray
    variance: np.ndarray
    pressure: np.ndarray


class IntervalSpace(SearchSpace):
    def bounded_margins(self, yes, no, lower, upper):
        unresolved = self.all ^ (yes | no)
        confirmed_count = yes.bit_count()
        yes_in_committee = self.pop[self.masks & yes]
        shared_capacity = self.pop[self.shared & unresolved]
        proposed_capacity = self.pop[self.first_only & unresolved]
        rival_capacity = self.pop[self.second_only & unresolved]
        outside_capacity = self.pop[self.outside & unresolved]
        maxima = np.full(self.shared.shape, -self.values[-1] - 1)
        for shared_count in range(self.seats + 1):
            fixed_count = confirmed_count + shared_count
            residual_lower = max(0, lower - fixed_count)
            residual_upper = np.minimum(
                upper - fixed_count, proposed_capacity + rival_capacity + outside_capacity
            )
            feasible = (shared_count <= shared_capacity) & (residual_lower <= residual_upper)
            rival_approvals = np.minimum(rival_capacity, residual_upper)
            proposed_approvals = np.maximum(0, residual_lower - rival_capacity - outside_capacity)
            proposed_index = np.clip(yes_in_committee[:, None] + shared_count + proposed_approvals, 0, self.seats)
            rival_index = np.clip(yes_in_committee + shared_count + rival_approvals, 0, self.seats)
            gap = self.values[rival_index] - self.values[proposed_index]
            maxima = np.maximum(maxima, np.where(feasible, gap, -self.values[-1] - 1))
        if np.any(maxima < -self.values[-1]):
            raise ValueError("Infeasible ballot interval")
        return maxima

    @cache
    def bounded_row(self, state):
        yes, no, lower, upper = state
        completions = self.possible(yes, no, -1)
        completions = completions[(self.pop[completions] >= lower) & (self.pop[completions] <= upper)]
        if not len(completions):
            raise ValueError("Infeasible ballot interval")
        return Row(completions, self.score[completions].mean(axis=0), self.bounded_margins(*state))

    def answer(self, ballot, question):
        if question < self.n_candidates:
            return bool(ballot & (1 << question))
        return int(ballot).bit_count() <= question - self.n_candidates

    def updated(self, state, question, answer):
        yes, no, lower, upper = state
        if question < self.n_candidates:
            mask = 1 << question
            if answer:
                yes |= mask
            else:
                no |= mask
        else:
            threshold = question - self.n_candidates
            if answer:
                upper = min(upper, threshold)
            else:
                lower = max(lower, threshold + 1)
        return yes, no, max(lower, yes.bit_count()), min(upper, self.n_candidates - no.bit_count())

    @cache
    def options(self, state):
        row = self.bounded_row(state)
        ids, variance, pressure = [], [], []
        for q in range(2 * self.n_candidates):
            matches = sum(self.answer(ballot, q) for ballot in row.completions)
            if not 0 < matches < len(row.completions):
                continue
            positive = self.bounded_row(self.updated(state, q, True))
            negative = self.bounded_row(self.updated(state, q, False))
            p = matches / len(row.completions)
            assert len(positive.completions) == matches
            assert len(negative.completions) == len(row.completions) - matches
            mean_diff = positive.averages - negative.averages
            ids.append(q)
            variance.append(p * (1 - p) * (mean_diff - mean_diff[:, None]) ** 2)
            pressure.append(row.margins - p * positive.margins - (1 - p) * negative.margins)
        shape = (-1, len(self.masks), len(self.masks))
        return Options(np.array(ids, dtype=int), np.array(variance).reshape(shape),
                       np.array(pressure).reshape(shape))


def reference_winner(space, ballots):
    return int(space.score[ballots].sum(axis=0).argmax())


def read_profile(space, ballots, seed, mode, mixed, threshold_cost):
    state = [(0, 0, 0, space.n_candidates)] * len(ballots)
    initial = space.bounded_row(state[0])
    expected = len(ballots) * initial.averages
    margins = len(ballots) * initial.margins
    ranks = np.random.default_rng(seed + 100000).permutation(len(ballots) * 2 * space.n_candidates)
    ranks = ranks.reshape(len(ballots), 2 * space.n_candidates)
    truth_scores = space.score[ballots].sum(axis=0)
    truth = reference_winner(space, ballots)
    cost = 0
    spent = None
    n_approval = n_threshold = 0
    cut = space.n_candidates
    for _ in range(len(ballots) * (1 << cut) + 1):
        leader = space.score_leader(expected)
        threats = margins[leader] + space.tie_cost[leader]
        if threats.max() == 0:
            assert leader == truth
            return {
                "approval_questions": n_approval,
                "threshold_questions": n_threshold,
                "price_total": cost / 2,
                "covered_by_budget": cost <= CHECKPOINT,
                "regret_at_budget": 0.0 if spent is None else spent,
            }
        if mode == "pressure":
            rivals = [j for j in np.argsort(-threats, kind="stable") if threats[j] > 0][:3]
            weights = 1 / (1 + (threats[rivals[0]] - threats[rivals]) / space.scale)
        else:
            rivals = [j for j in np.argsort(-expected, kind="stable") if j != leader][:3]
            weights = 1 / (1 + np.maximum(0, expected[leader] - expected[rivals]) / space.scale)
        best = (-np.inf, np.inf, None)
        for i, current in enumerate(state):
            options = space.options(current)
            n_allowed = len(options.ids) if mixed else int(np.searchsorted(options.ids, cut))
            if not n_allowed:
                continue
            allowed_ids = options.ids[:n_allowed]
            prices = np.where(allowed_ids < cut, 2, threshold_cost)
            info = getattr(options, mode)[:n_allowed, leader, :][:, rivals] @ weights / prices
            max_info = float(info.max())
            within = np.flatnonzero(info >= max_info - 1e-12)
            j = int(within[np.argmin(ranks[i, allowed_ids[within]])])
            priority = int(ranks[i, allowed_ids[j]])
            if max_info > best[0] + 1e-12 or (abs(max_info - best[0]) <= 1e-12 and priority < best[1]):
                best = (max_info, priority, (i, int(allowed_ids[j]), int(prices[j])))
        assert best[2] is not None
        voter, question, price = best[2]
        if spent is None and cost + price > CHECKPOINT:
            spent = int(truth_scores[truth] - truth_scores[leader]) / space.scale
        before = space.bounded_row(state[voter])
        state[voter] = space.updated(state[voter], question, space.answer(ballots[voter], question))
        after = space.bounded_row(state[voter])
        assert len(after.completions) < len(before.completions) and ballots[voter] in after.completions
        change = after.margins - before.margins
        assert change.max() == 0
        margins += change
        expected += after.averages - before.averages
        n_approval += question < cut
        n_threshold += question >= cut
        cost += price
    raise RuntimeError("Certificate failed to settle after fully revealed ballots")


def generated(seed, family):
    rng = np.random.default_rng(seed)
    utility = rng.normal(size=(VOTERS, PROPOSALS))
    if family in ("clustered", "polarised"):
        clustered = family == "clustered"
        factions = rng.choice(2, size=VOTERS, p=(0.60, 0.40) if clustered else (0.5, 0.5))
        boost = 2.3 if clustered else 3.6
        utility[np.arange(VOTERS), factions * 2] += boost
        utility[np.arange(VOTERS), factions * 2 + 1] += boost
        utility[:, -1] += 0.70 if clustered else 0.20
    if family == "near_consensus":
        utility[:, :2] += 2.3
    lengths = rng.choice((1, 2, 3), size=VOTERS, p=(0.26, 0.46, 0.28))
    relabel = rng.permutation(PROPOSALS)
    return np.array([
        sum(1 << int(relabel[candidate]) for candidate in np.argsort(values)[-length:])
        for values, length in zip(utility, lengths, strict=True)
    ], dtype=int)


def projected(seed, projects, approvals):
    rng = np.random.default_rng(seed)
    choices = [projects[index] for index in rng.choice(len(projects), PROPOSALS, replace=False)]
    return np.array([
        sum(1 << bit for bit, project in enumerate(choices) if project in approvals[index])
        for index in rng.choice(len(approvals), VOTERS, replace=False)
    ], dtype=int)


def audit():
    tiny = IntervalSpace(4, 2)
    feasible_states = []
    for flags in product(range(3), repeat=4):
        yes = sum(1 << j for j, observed in enumerate(flags) if observed == 1)
        no = sum(1 << j for j, observed in enumerate(flags) if observed == 2)
        for lo in range(yes.bit_count(), 5 - no.bit_count()):
            for hi in range(lo, 5 - no.bit_count()):
                state = (yes, no, lo, hi)
                row = tiny.bounded_row(state)
                exhaustive = tiny.score[row.completions]
                assert np.array_equal(row.margins, (exhaustive[:, None, :] - exhaustive[:, :, None]).max(axis=0))
                feasible_states.append(state)
                if lo == hi:
                    assert np.array_equal(row.margins, tiny.max_margins(yes, no, lo))

    rng = np.random.default_rng(400510)
    completion_count = 0
    for _ in range(75):
        rows = [tiny.bounded_row(feasible_states[j]) for j in rng.choice(len(feasible_states), size=3)]
        worlds = np.asarray(list(product(*(row.completions for row in rows))))
        completion_count += len(worlds)
        all_scores = tiny.score[worlds].sum(axis=1)
        all_winners = all_scores.argmax(axis=1)
        max_margins = sum(row.margins for row in rows)
        assert np.array_equal(max_margins, (all_scores[:, None, :] - all_scores[:, :, None]).max(axis=0))
        expected_candidate = tiny.score_leader(sum(row.averages for row in rows))
        certificate = (max_margins[expected_candidate] + tiny.tie_cost[expected_candidate]).max() == 0
        assert certificate == bool(np.all(all_winners == expected_candidate))

    space = IntervalSpace(PROPOSALS, SEATS)
    witness = np.ones(VOTERS, dtype=int)
    incumbent = reference_winner(space, witness)
    assert space.committees[incumbent] == tuple(range(SEATS))
    for v, c in product(range(VOTERS), range(SEATS, PROPOSALS)):
        alternative = witness.copy()
        alternative[v] |= 1 << c
        assert reference_winner(space, alternative) != incumbent
    for voter, question in product(range(VOTERS), range(2 * PROPOSALS)):
        single = space.updated((0, 0, 0, PROPOSALS), question, space.answer(1, question))
        completions = space.bounded_row(single).completions
        profiles = np.tile(witness, (len(completions), 1))
        profiles[:, voter] = completions
        winners = space.score[profiles].sum(axis=1).argmax(axis=1)
        assert np.any(winners != incumbent)
    answer = space.updated((0, 0, 0, PROPOSALS), PROPOSALS + 1, True)
    answer = space.updated(answer, 0, True)
    row = space.bounded_row(answer)
    assert np.array_equal(row.completions, [1])
    assert (VOTERS * row.margins + space.tie_cost)[incumbent].max() == 0
    return {
        "interval_states_exhaustively_verified": len(feasible_states),
        "tiny_profile_completions": completion_count,
        "individual_approval_lower_bound": VOTERS * (PROPOSALS - SEATS),
        "single_question_counterexamples_verified": VOTERS * 2 * PROPOSALS,
        "mixed_interface_minimum_on_witness": min(VOTERS * (PROPOSALS - SEATS), 2 * VOTERS),
    }


def analyse(records, bit, mixed):
    individual = np.array([row[bit]["price_total"] for row in records])
    expanded = np.array([row[mixed]["price_total"] for row in records])
    difference = individual - expanded
    samples = np.random.default_rng(428200).integers(0, len(records), (5000, len(records)))
    interval = np.quantile(difference[samples].mean(axis=1), (0.025, 0.975))
    return {
        "number": len(records),
        "mean_individual_cost": float(individual.mean()),
        "mean_mixed_cost": float(expanded.mean()),
        "mean_saving": float(difference.mean()),
        "mean_fraction_saved": float(difference.mean() / individual.mean()),
        "profile_bootstrap_95": interval.tolist(),
        "mean_threshold_questions": float(np.mean([row[mixed]["threshold_questions"] for row in records])),
        "coverage_at_thirty": {
            mode: float(np.mean([row[key]["covered_by_budget"] for row in records]))
            for mode, key in (("individual", bit), ("mixed", mixed))
        },
        "mean_pav_regret_at_thirty": {
            mode: float(np.mean([row[key]["regret_at_budget"] for row in records]))
            for mode, key in (("individual", bit), ("mixed", mixed))
        },
    }


def two_interfaces(space, ballots, seed):
    return {f"{mode}_{interface}": read_profile(space, ballots, seed, mode, interface == "mixed", 2)
            for mode, interface in product(SELECTORS, ("individual", "mixed"))}


def run(data_dir):
    protocol = Path(__file__).with_name("BINARY_QUERY_PROTOCOL.md")
    assert digest(protocol.read_bytes()) == PROTOCOL_SHA256
    space = IntervalSpace(PROPOSALS, SEATS)
    audit_results = audit()
    dev_results = {kind: [] for kind in SELECTORS}
    for seed in range(421200, 421248):
        ballots = generated(seed, "clustered")
        for selector in SELECTORS:
            dev_results[selector].append(read_profile(space, ballots, seed, selector, False, 2)["price_total"])
    development_means = {name: float(np.mean(values)) for name, values in dev_results.items()}
    champion = min(development_means, key=development_means.get)
    bit, mixed = f"{champion}_individual", f"{champion}_mixed"

    synthetic = {}
    for family, (start, size) in PANELS.items():
        records = []
        for seed in range(start, start + size):
            ballots = generated(seed, family)
            row = two_interfaces(space, ballots, seed)
            if family == "clustered":
                for cost in (3, 4):
                    row[f"{mixed}_{cost / 2}"] = read_profile(space, ballots, seed, champion, True, cost)
            records.append({"seed": seed, **row})
        synthetic[family] = {
            "summaries": {method: analyse(records, f"{method}_individual", f"{method}_mixed") for method in SELECTORS},
            "records": records,
        }
        if family == "clustered":
            synthetic[family]["price_sensitivity"] = {
                str(cost / 2): analyse(records, bit, f"{mixed}_{cost / 2}") for cost in (3, 4)
            }
        print(f"Completed {family} ({size} profiles).", flush=True)

    external = {}
    excluded = []
    for index, (filename, expected_hash) in enumerate(SOURCES):
        archive = (data_dir / filename).read_bytes()
        assert digest(archive) == expected_hash
        _, projects, approvals = split_voting_data(archive)
        if not projects:
            excluded.append(filename)
            continue
        records = []
        for seed in range(426200 + 100 * index, 426220 + 100 * index):
            ballots = projected(seed, projects, approvals)
            row = two_interfaces(space, ballots, seed)
            records.append({"seed": seed, **row})
        summaries = {method: analyse(records, f"{method}_individual", f"{method}_mixed") for method in SELECTORS}
        # Samples are repeated projections from one source. Keep external summaries descriptive.
        for item in summaries.values():
            item.pop("profile_bootstrap_95")
        external[filename] = {"summaries": summaries, "records": records}
        print(f"Completed {filename}.", flush=True)
    primary = synthetic["clustered"]["summaries"][champion]
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "audit": audit_results,
        "development_bit_only_mean": development_means,
        "primary_selector": champion,
        "primary": primary,
        "primary_gate_passed": bool(primary["mean_fraction_saved"] >= 0.05 and primary["profile_bootstrap_95"][0] > 0),
        "synthetic": synthetic,
        "external": external,
        "external_source_macro_saving": float(np.mean([
            dataset["summaries"][champion]["mean_saving"] for dataset in external.values()
        ])),
        "external_excluded": excluded,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    paths = parser.add_mutually_exclusive_group(required=True)
    paths.add_argument("--audit-only", action="store_true")
    paths.add_argument("--output", type=Path)
    paths.add_argument("--verify", type=Path)
    opts = parser.parse_args()
    if opts.audit_only:
        print(json.dumps(audit(), indent=2))
        return
    if opts.data_dir is None:
        parser.error("--data-dir is required for the empirical run")
    payload = json.dumps(run(opts.data_dir), sort_keys=True, indent=2, allow_nan=False) + "\n"
    if opts.verify:
        assert opts.verify.read_text() == payload, "Mismatched binary elicitation replay"
        print("Verified binary elicitation results byte-for-byte.")
    else:
        opts.output.write_text(payload)


if __name__ == "__main__":
    main()
