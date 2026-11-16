"""Evaluate two-question certificate-pressure planning for proportional approval voting."""

import argparse
from functools import cache
from itertools import product
import json
from pathlib import Path

import numpy as np

from binary_queries import (
    IntervalSpace,
    PROPOSALS,
    SEATS,
    generated,
    projected,
    read_profile,
    reference_winner,
)
from voting import SOURCES, digest, split_voting_data


PROTOCOL_SHA256 = "009588da49e997b761aa5b92a74d1e656c1100fe7a8611778d6b2e705ec081fb"
PANELS = {
    "clustered": (712100, 120),
    "independent": (712300, 36),
    "polarised": (712400, 36),
    "near_consensus": (712500, 36),
}
POLICIES = tuple(product((1, 2), ("individual", "mixed")))


class LookaheadSpace(IntervalSpace):
    @cache
    def children(self, state, question):
        current = self.bounded_row(state)
        yes_state = self.updated(state, question, True)
        no_state = self.updated(state, question, False)
        chance_yes = len(self.bounded_row(yes_state).completions) / len(current.completions)
        assert 0 < chance_yes < 1
        return yes_state, no_state, chance_yes

    def action_values(self, states, leader, rivals, weights, horizon, mixed):
        """Find the best conditional second question under fixed linear weights."""
        cut = 2 * self.n_candidates if mixed else self.n_candidates

        @cache
        def information(state):
            options = self.options(state)
            size = int(np.searchsorted(options.ids, cut))
            gains = options.pressure[:size, leader, :][:, rivals] @ weights
            return options.ids[:size], gains, max(0.0, float(gains.max(initial=0)))

        rows = [information(state) for state in states]
        if horizon == 1:
            return [(questions, gains) for questions, gains, _ in rows]

        # If the first query touches one row, all gains on other rows are unchanged.
        by_best_gain = sorted(range(len(rows)), key=lambda i: -rows[i][2])
        top = by_best_gain[0]
        top_other = rows[by_best_gain[1]][2] if len(by_best_gain) > 1 else 0.0
        calculated = {}
        for voter, state in enumerate(states):
            other = top_other if voter == top else rows[top][2]
            key = (state, other)
            if key not in calculated:
                questions, gains, _ = rows[voter]
                continuation = []
                for question in questions:
                    yes, no, chance_yes = self.children(state, int(question))
                    best_yes = max(other, information(yes)[2])
                    best_no = max(other, information(no)[2])
                    continuation.append(chance_yes * best_yes + (1 - chance_yes) * best_no)
                calculated[key] = questions, gains + continuation
        return [
            calculated[state, top_other if voter == top else rows[top][2]]
            for voter, state in enumerate(states)
        ]


def play(space, ballots, seed, horizon, interface):
    cut = space.n_candidates
    states = [(0, 0, 0, cut)] * len(ballots)
    initial = space.bounded_row(states[0])
    expected = len(ballots) * initial.averages
    margins = len(ballots) * initial.margins
    ranks = np.random.default_rng(seed + 100000).permutation(len(ballots) * 2 * cut)
    ranks = ranks.reshape(len(ballots), 2 * cut)
    truth = reference_winner(space, ballots)
    approval = threshold = 0
    for _ in range(len(ballots) * (1 << cut) + 1):
        leader = space.score_leader(expected)
        threats = margins[leader] + space.tie_cost[leader]
        if threats.max() == 0:
            assert leader == truth
            return {
                "approval_questions": approval,
                "threshold_questions": threshold,
                "questions": approval + threshold,
                "covered_by_30": approval + threshold <= 30,
                "correct": bool(leader == truth),
            }
        rivals = [j for j in np.argsort(-threats, kind="stable") if threats[j] > 0][:3]
        weights = 1 / (1 + (threats[rivals[0]] - threats[rivals]) / space.scale)
        options = space.action_values(states, leader, rivals, weights, horizon, interface == "mixed")
        best = (-np.inf, np.inf, None)
        for voter, (questions, values) in enumerate(options):
            if not len(questions):
                continue
            value = float(values.max())
            within = np.flatnonzero(values >= value - 1e-12)
            selected = int(questions[within[np.argmin(ranks[voter, questions[within]])]])
            priority = int(ranks[voter, selected])
            if value > best[0] + 1e-12 or (abs(value - best[0]) <= 1e-12 and priority < best[1]):
                best = value, priority, (voter, selected)
        assert best[2] is not None
        voter, question = best[2]
        before = space.bounded_row(states[voter])
        states[voter] = space.updated(states[voter], question, space.answer(ballots[voter], question))
        after = space.bounded_row(states[voter])
        assert len(after.completions) < len(before.completions) and ballots[voter] in after.completions
        change = after.margins - before.margins
        assert change.max() == 0
        expected += after.averages - before.averages
        margins += change
        approval += question < cut
        threshold += question >= cut
    raise RuntimeError("Every approval was revealed without a certified winner")


def direct_leaf_evaluation(space, rows, leader, rivals, weights, horizon, mixed):
    """Independent oracle that enumerates row completions and both question depths."""
    cut = space.n_candidates
    ballots = [tuple(int(ballot) for ballot in row) for row in rows]

    def answers(ballot, q):
        if q < cut:
            return (ballot >> q) & 1
        return ballot.bit_count() <= q - cut

    def branches(current):
        for voter, row in enumerate(current):
            for q in range(2 * cut if mixed else cut):
                yes = tuple(ballot for ballot in row if answers(ballot, q))
                no = tuple(ballot for ballot in row if not answers(ballot, q))
                if yes and no:
                    yield (voter, q), (
                        (len(group) / len(row), (*current[:voter], group, *current[voter + 1:]))
                        for group in (yes, no)
                    )

    def cost(current):
        return sum(
            weights[slot] * sum(
                max(int(space.score[ballot, rival] - space.score[ballot, leader]) for ballot in row)
                for row in current
            )
            for slot, rival in enumerate(rivals)
        )

    def second_cost(current):
        return min(
            (sum(prob * cost(child) for prob, child in children) for _, children in branches(current)),
            default=cost(current),
        )

    leaf = cost if horizon == 1 else second_cost
    before = cost(ballots)
    return {
        action: before - sum(prob * leaf(child) for prob, child in children)
        for action, children in branches(ballots)
    }


def audit(space):
    tiny = LookaheadSpace(3, 2)
    empty = (0, 0, 0, 3)
    partial = (1, 0, 1, 3)
    at_most_one = (0, 0, 0, 1)
    known = (1, 6, 1, 1)
    fixtures = (
        (empty,),
        (empty, at_most_one),
        (partial, at_most_one, empty),
        (known, empty, partial),
        (known, known),
    )
    comparisons = 0
    for states in fixtures:
        completions = [tiny.bounded_row(state).completions for state in states]
        for leader in range(len(tiny.masks)):
            rivals = [i for i in range(len(tiny.masks)) if i != leader]
            weights = np.array([1.0, 0.5])
            for horizon, interface in POLICIES:
                mixed = interface == "mixed"
                proposed = {
                    (voter, int(q)): float(value)
                    for voter, (q_ids, values) in enumerate(
                        tiny.action_values(states, leader, rivals, weights, horizon, mixed)
                    )
                    for q, value in zip(q_ids, values, strict=True)
                }
                reference = direct_leaf_evaluation(tiny, completions, leader, rivals, weights, horizon, mixed)
                assert proposed.keys() == reference.keys()
                assert all(abs(value - reference[key]) < 1e-10 for key, value in proposed.items())
                comparisons += len(proposed)
    for seed in range(712000, 712008):
        ballots = generated(seed, "clustered")
        for interface in ("individual", "mixed"):
            original = read_profile(space, ballots, seed, "pressure", interface == "mixed", 2)
            replay = play(space, ballots, seed, 1, interface)
            for key in ("approval_questions", "threshold_questions"):
                assert original[key] == replay[key]
            assert original["price_total"] == replay["questions"]
    return {
        "development_profiles": 8,
        "one_step_matches": 16,
        "exhaustive_tiny_action_value_checks": comparisons,
    }


def profile_row(space, ballots, seed):
    output = {"seed": seed}
    for horizon, interface in POLICIES:
        output[f"{horizon}_{interface}"] = play(space, ballots, seed, horizon, interface)
    return output


def summarise(records, inference):
    keys = [f"{steps}_{interface}" for steps, interface in POLICIES]
    questions = {key: np.array([row[key]["questions"] for row in records]) for key in keys}
    thresholds = {key: np.array([row[key]["threshold_questions"] for row in records]) for key in keys}
    one_step = questions["1_individual"] - questions["1_mixed"]
    two_step = questions["2_individual"] - questions["2_mixed"]
    contrasts = {
        "two_step_interface": two_step,
        "one_step_interface": one_step,
        "interaction": two_step - one_step,
        "individual_horizon": questions["1_individual"] - questions["2_individual"],
        "mixed_horizon": questions["1_mixed"] - questions["2_mixed"],
    }
    samples = (
        np.random.default_rng(712901).integers(0, len(records), (5000, len(records)))
        if inference else None
    )
    effects = {}
    for name, difference in contrasts.items():
        result = {"mean_saving": float(difference.mean())}
        if samples is not None:
            result["profile_bootstrap_95"] = np.quantile(
                difference[samples].mean(axis=1), (0.025, 0.975)
            ).tolist()
        effects[name] = result
    return {
        "number": len(records),
        "policies": {
            key: {
                "mean_questions": float(questions[key].mean()),
                "mean_threshold_questions": float(thresholds[key].mean()),
                "error_count": sum(not row[key]["correct"] for row in records),
                "coverage_by_30": float(np.mean([row[key]["covered_by_30"] for row in records])),
            }
            for key in keys
        },
        "comparisons": effects,
        "fraction_saved_two_step": float(two_step.mean() / questions["2_individual"].mean()),
        "fixed_policy_savings": {
            str(price): {
                "one_step": float((one_step - (price - 1) * thresholds["1_mixed"]).mean()),
                "two_step": float((two_step - (price - 1) * thresholds["2_mixed"]).mean()),
            }
            for price in (1.5, 2.0)
        },
    }


def run(data_dir):
    protocol = Path(__file__).with_name("LOOKAHEAD_PROTOCOL.md")
    assert digest(protocol.read_bytes()) == PROTOCOL_SHA256
    space = LookaheadSpace(PROPOSALS, SEATS)
    checks = audit(space)
    synthetic = {}
    for family, (start, count) in PANELS.items():
        records = [
            profile_row(space, generated(seed, family), seed)
            for seed in range(start, start + count)
        ]
        synthetic[family] = {"summary": summarise(records, True), "records": records}
        print(f"Completed {family} ({count} profiles).", flush=True)

    external = {}
    excluded = []
    for index, (name, expected_hash) in enumerate(SOURCES):
        archive = (data_dir / name).read_bytes()
        assert digest(archive) == expected_hash, name
        _, projects, approvals = split_voting_data(archive)
        if not projects:
            excluded.append(name)
            continue
        start = 713000 + 100 * index
        records = [
            profile_row(space, projected(seed, projects, approvals), seed)
            for seed in range(start, start + 20)
        ]
        external[name] = {"summary": summarise(records, False), "records": records}
        print(f"Completed {name} ({len(records)} profiles).", flush=True)

    primary = synthetic["clustered"]["summary"]
    interval = primary["comparisons"]["two_step_interface"]["profile_bootstrap_95"]
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "audit": checks,
        "primary_gate_passed": bool(primary["fraction_saved_two_step"] >= 0.05 and interval[0] > 0),
        "synthetic": synthetic,
        "external": external,
        "external_excluded": excluded,
        "external_source_macro_savings": {
            name: float(np.mean([
                dataset["summary"]["comparisons"][name]["mean_saving"]
                for dataset in external.values()
            ])) for name in primary["comparisons"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--audit-only", action="store_true")
    outputs.add_argument("--output", type=Path)
    outputs.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit(LookaheadSpace(PROPOSALS, SEATS)), indent=2))
        return
    if args.data_dir is None:
        parser.error("--data-dir is required to run the frozen experiment")
    report = json.dumps(run(args.data_dir), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.verify:
        assert report == args.verify.read_text(), "Byte-for-byte replay mismatch"
        print("Verified lookahead results byte-for-byte.")
    else:
        args.output.write_text(report)


if __name__ == "__main__":
    main()
