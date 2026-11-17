"""Sharp certificates for PAV winners given incomplete and potentially wrong reports."""
import argparse
from hashlib import sha256
from itertools import product
import json
from math import lcm
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voting import SOURCES, SearchSpace, split_voting_data

BUDGET = 3
IMPOSSIBLE = -1_000_000
VOTERS = 36


class RobustTable:
    """Precompute row tables so repeated replays need only short budget convolutions."""

    def __init__(self, candidate_count, seats):
        self.space = SearchSpace(candidate_count, seats)
        self.ballots = self.space.ballots
        self.gaps = self.space.score[:, None, :] - self.space.score[:, :, None]
        size = len(self.space.masks)
        states = 3 ** candidate_count
        self.size = size
        self.values = np.full((states, BUDGET + 1, size, size), IMPOSSIBLE, dtype=np.int64)
        self.argmax = np.full(self.values.shape, -1, dtype=np.int16)
        self.means = np.zeros((states, size))
        self.state_id = {}
        for i, counts in enumerate(product(range(3), repeat=candidate_count)):
            mask = sum(int(value != 0) << j for j, value in enumerate(counts))
            yes = sum(int(value == 2) << j for j, value in enumerate(counts))
            self.state_id[mask, yes] = i
            disagreement = self.space.pop[(self.ballots ^ yes) & mask]
            self.means[i] = self.space.score[disagreement == 0].mean(axis=0)
            for cost in range(min(mask.bit_count(), BUDGET) + 1):
                choices = self.ballots[disagreement == cost]
                leaders = self.gaps[choices].argmax(axis=0)
                self.argmax[i, cost] = choices[leaders]
                self.values[i, cost] = np.take_along_axis(
                    self.gaps[choices], leaders[None, :, :], axis=0
                )[0]
        self.denominator = lcm(*range(1, BUDGET + 1))
        self.slopes = np.array(sorted({
            (self.denominator // cost) * rise
            for rise in range(1 + 2 * int(self.space.values[-1]))
            for cost in range(1, BUDGET + 1)
        }))
        self.slope_cost = self.slopes * np.arange(BUDGET + 1)[:, None]

    def observations(self, masks, ballots):
        return np.array([self.state_id[int(mask), int(mask & ballot)] for mask, ballot in zip(masks, ballots)])

    def optimise(self, states, leader):
        values = self.values[states, :, leader]
        scratch = np.zeros((BUDGET + 1, self.size), dtype=np.int64)
        choices = np.zeros((len(states), BUDGET + 1, self.size), dtype=np.int16)
        for row, scores in enumerate(values):
            options = np.full((BUDGET + 1, BUDGET + 1, self.size), IMPOSSIBLE)
            for cost in range(BUDGET + 1):
                options[cost, cost:] = scratch[:BUDGET + 1 - cost] + scores[cost]
            choices[row] = options.argmax(axis=0)
            scratch = options.max(axis=0)
        return scratch, choices

    def witness(self, states, leader, rival, budget, choices):
        completion = []
        for i in range(len(states) - 1, -1, -1):
            cost = choices[i, budget, rival]
            completion.append(int(self.argmax[states[i], cost, leader, rival]))
            budget -= cost
        return tuple(reversed(completion))

    def evaluate(self, masks, ballots):
        states = self.observations(masks, ballots)
        leader = self.space.score_leader(self.means[states].sum(axis=0))
        exact, choices = self.optimise(states, leader)
        values = self.values[states, :, leader]
        dual = (values[..., None] * self.denominator - self.slope_cost[None, :, None, :]).max(axis=1).sum(axis=0)
        fraction = (dual[None, :, :] + np.arange(BUDGET + 1)[:, None, None] * self.slopes).min(axis=2)
        fraction //= self.denominator
        trivial = exact[0] + self.space.scale * np.arange(BUDGET + 1)[:, None]
        trivial[:, leader] = 0  # The target committee has identically zero self-margin.
        assert np.all(exact <= fraction) and np.all(fraction <= trivial)
        assert np.all(exact[0] == fraction[0])
        assert np.all(exact[1:] >= exact[:-1])
        tie_cost = (np.arange(self.size) < leader).astype(int)
        bad = exact + tie_cost > 0
        result = {
            "exact": (~bad.any(axis=1)).tolist(),
            "fractional": (fraction + tie_cost <= 0).all(axis=1).tolist(),
            "lipschitz": (trivial + tie_cost <= 0).all(axis=1).tolist(),
        }
        observed_winner = int(self.space.score[ballots].sum(axis=0).argmax())
        if result["exact"][0]:
            assert leader == observed_winner
        failures = np.flatnonzero(bad.any(axis=1))
        witness = None
        if len(failures):
            budget = int(failures[0])
            rival = int(np.flatnonzero(bad[budget])[0])
            completion = self.witness(states, leader, rival, budget, choices)
            score = self.space.score[np.asarray(completion)].sum(axis=0)
            distance = sum(((a ^ b) & int(mask)).bit_count()
                           for a, b, mask in zip(map(int, ballots), completion, masks))
            assert distance == budget and int(score[rival] - score[leader]) == exact[budget, rival]
            assert int(score.argmax()) != leader
            witness = {"budget": budget, "rival": rival, "gap_units": int(exact[budget, rival]),
                       "alternate_winner": int(score.argmax())}
        return {"leader": leader, "certificate": result, "first_failure": witness}


def mask_for(rng, rows, columns=6, rate=.90):
    masks = np.zeros(rows, dtype=int)
    for i in rng.permutation(rows * columns)[:int(rate * rows * columns)]:
        masks[i // columns] |= 1 << (i % columns)
    return masks


def synthetic(seed, family):
    rng = np.random.default_rng(seed)
    utility = rng.normal(size=(VOTERS, 6))
    if family == "clustered":
        factions = rng.choice(3, size=VOTERS, p=(.49, .31, .20))
        utility[np.arange(VOTERS), 2 * factions] += 2.3
        utility[np.arange(VOTERS), 2 * factions + 1] += 2.3
    length = rng.choice((1, 2, 3, 4), size=VOTERS, p=(.18, .38, .32, .12))
    labels = rng.permutation(6)
    return np.array([
        sum(1 << int(labels[candidate]) for candidate in np.argsort(row)[-size:])
        for row, size in zip(utility, length)
    ])


def historical(seed, projects, approvals):
    rng = np.random.default_rng(seed)
    columns = rng.choice(len(projects), 6, replace=False)
    keys = tuple(projects[i] for i in columns)
    voters = rng.choice(len(approvals), VOTERS, replace=False)
    return np.array([sum((key in approvals[i]) << j for j, key in enumerate(keys)) for i in voters])


def exhaustive_audit():
    """Recompute every winner and difference independently of row tables."""
    table = RobustTable(3, 2)
    rng = np.random.default_rng(4190088)
    ballots_all = list(product(range(8), repeat=3))
    checks = 0
    for _ in range(48):
        ballots = rng.integers(0, 8, size=3)
        masks = rng.integers(0, 8, size=3)
        states = table.observations(masks, ballots)
        by_budget = [[] for _ in range(BUDGET + 1)]
        for candidate in ballots_all:
            distance = sum(((int(actual) ^ trial) & int(mask)).bit_count()
                           for actual, trial, mask in zip(ballots, candidate, masks))
            score_vector = tuple(
                sum((0, 2, 3)[sum((ballot >> player) & 1 for player in committee)] for ballot in candidate)
                for committee in table.space.committees)
            for b in range(distance, BUDGET + 1):
                by_budget[b].append(score_vector)
        for leader in range(table.size):
            exact, choices = table.optimise(states, leader)
            for b, scores in enumerate(by_budget):
                for rival in range(table.size):
                    max_gap = max(score[rival] - score[leader] for score in scores)
                    assert exact[b, rival] == max_gap
                    witness = table.witness(states, leader, rival, b, choices)
                    assert sum(((int(actual) ^ trial) & int(mask)).bit_count()
                               for actual, trial, mask in zip(ballots, witness, masks)) <= b
                    witness_scores = table.space.score[list(witness)].sum(axis=0)
                    assert witness_scores[rival] - witness_scores[leader] == max_gap
                    checks += 1
        result = table.evaluate(masks, ballots)
        cert = [all(score.index(max(score)) == result["leader"] for score in choices)
                for choices in by_budget]
        assert result["certificate"]["exact"] == cert
    return checks


def strict_fixture(table):
    """Find a small, deliberately selected separation independent of the main panels."""
    rng = np.random.default_rng(4191149)
    for i in range(20_000):
        ballots = rng.integers(0, 64, size=6)
        masks = mask_for(rng, 6)
        report = table.evaluate(masks, ballots)
        cert = report["certificate"]
        if cert["exact"][2] and not cert["fractional"][2]:
            return {"search_position": i, "masks": masks.tolist(), "reported_yes": (masks & ballots).tolist(),
                    "leader": report["leader"], "certificates": cert}
    return None


def profile(table, ballots, seed):
    mask = mask_for(np.random.default_rng(seed + 19000), len(ballots))
    return {"seed": seed, **table.evaluate(mask, ballots)}


def summarise(rows):
    by_policy = {key: np.asarray([r["certificate"][key] for r in rows], dtype=bool)
                 for key in ("exact", "fractional", "lipschitz")}
    exact = by_policy["exact"]
    return {
        "profiles": len(rows),
        "certifications_by_budget": {key: value.sum(axis=0).tolist() for key, value in by_policy.items()},
        "zero_error_certs_broken_by_budget": (exact[:, [0]] & ~exact).sum(axis=0).tolist(),
        "additional_vs_fractional_by_budget": (exact & ~by_policy["fractional"]).sum(axis=0).tolist(),
        "additional_vs_lipschitz_by_budget": (exact & ~by_policy["lipschitz"]).sum(axis=0).tolist(),
    }


def run(data_dir):
    checks = exhaustive_audit()
    table = RobustTable(6, 3)
    fixture = strict_fixture(table)
    panels = {
        family: [profile(table, synthetic(seed, family), seed) for seed in range(start, start + count)]
        for family, start, count in (("clustered", 4191270, 160), ("independent", 4192270, 80))
    }
    archived = {}
    for source_index, (file, expected) in enumerate(SOURCES):
        contents = (data_dir / file).read_bytes()
        assert sha256(contents).hexdigest() == expected
        _, projects, approvals = split_voting_data(contents)
        if approvals:
            assert len(approvals) >= VOTERS
            start = 4193270 + 100 * source_index
            archived[file] = [profile(table, historical(seed, projects, approvals), seed)
                              for seed in range(start, start + 24)]
    assert len(archived) == 5
    primary = np.array([int(r["certificate"]["exact"][2]) - int(r["certificate"]["fractional"][2])
                        for r in panels["clustered"]])
    draws = np.random.default_rng(4198860).integers(0, len(primary), size=(4000, len(primary)))
    interval = np.quantile(primary[draws].mean(axis=1), [.025, .975]).tolist()
    return {
        "protocol_sha256": sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest(),
        "audit": {"independently_enumerated_margin_checks": checks, "strict_fractional_separation": fixture},
        "primary": {"coverage_gain": float(primary.mean()), "paired_profile_95_interval": interval,
                    "practical_threshold_met": bool(primary.mean() >= .03 and interval[0] > 0)},
        "synthetic": {key: {"summary": summarise(value), "records": value} for key, value in panels.items()},
        "historical": {key: {"summary": summarise(value), "records": value} for key, value in archived.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = run(args.data_dir)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        assert text == args.verify.read_text(), "Robustness replay differs"
    print(json.dumps({"primary": result["primary"], "audit": result["audit"],
                      **{name: {key: row["summary"] for key, row in result[name].items()}
                         for name in ("synthetic", "historical")}}, indent=2))
