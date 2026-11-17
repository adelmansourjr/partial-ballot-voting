"""Independent exhaustive one-bit votes and scoped incentive checks."""
import argparse
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from math import comb
from pathlib import Path
from statistics import fmean

import numpy as np

from robust import RobustTable, SOURCES, VOTERS, historical, mask_for, split_voting_data, synthetic


def pair_separation(table):
    """A constructive lower bound on the fractional gap and exhaustive integer check."""
    ballots = (31, 47, 55)
    target = table.space.committees.index((0, 1, 2))
    rival = table.space.committees.index((3, 4, 5))
    masks = (63, 63, 63)
    states = table.observations(masks, ballots)
    exact, _ = table.optimise(states, target)

    def direct_gap(profile):
        def score(committee):
            return sum((0, 6, 9, 11)[sum((ballot >> i) & 1 for i in committee)] for ballot in profile)
        return score((3, 4, 5)) - score((0, 1, 2))

    alternative_gaps = []
    for count in range(3):
        for changed in combinations(range(18), count):
            altered = list(ballots)
            for bit in changed:
                altered[bit // 6] ^= 1 << (bit % 6)
            alternative_gaps.append(direct_gap(altered))
    assert len(alternative_gaps) == 172
    assert max(alternative_gaps) == int(exact[2, rival]) == -1
    row_steps = table.values[states, :, target, rival]
    assert np.array_equal(row_steps, np.tile((-2, 0, 3, 9), (3, 1)))
    # All slopes of chord from zero to each endpoint are <= 11/3. The
    # bound is attained by accepting a cost-3 edit with mass 2/3.
    for row in row_steps:
        assert all(Fraction(int(row[c] - row[0]), c) <= Fraction(11, 3) for c in (1, 2, 3))
    best_fractional_gap = Fraction(2, 3) * int(row_steps[0, 3] - row_steps[0, 0]) + direct_gap(ballots)
    assert best_fractional_gap == Fraction(4, 3)
    return {"ballots": ballots, "target": "ABC", "rival": "DEF", "scale": table.space.scale,
            "states_independently_enumerated": len(alternative_gaps), "integer_margin_units": -1,
            "fractional_margin_units": str(best_fractional_gap)}


def trace(table, ballots, seed):
    masks = np.full(len(ballots), 63)
    full = table.evaluate(masks, ballots)
    partial = table.evaluate(mask_for(np.random.default_rng(seed + 19000), VOTERS, rate=.98), ballots)
    row = table.space.score[ballots]
    score = row.sum(axis=0)
    winner = int(score.argmax())
    changed_rows = table.space.score[ballots[:, None] ^ table.space.cells]
    after = score - row[:, None, :] + changed_rows
    chosen = after.argmax(axis=-1)
    changed = chosen != winner
    assert bool(changed.any()) == (not full["certificate"]["exact"][1])
    assert full["leader"] == winner
    original_utility = table.space.pop[ballots & table.space.masks[winner]]
    altered_utility = table.space.pop[ballots[:, None] & table.space.masks[chosen]]
    improved = altered_utility > original_utility[:, None]
    assert np.all(improved <= changed)
    original_unique = int((score == score.max()).sum()) == 1
    alternate_unique = (after == after.max(axis=-1, keepdims=True)).sum(axis=-1) == 1
    strict = improved & alternate_unique & original_unique
    count = int(changed.sum())
    assert changed.size == 216
    detection = 1 - comb(216 - count, 20) / comb(216, 20)
    return {"seed": seed, "complete_certificates": full["certificate"], "partial_certificates": partial["certificate"],
            "changed_bits": count, "profitable_bits": int(improved.sum()), "unique_profitable_bits": int(strict.sum()),
            "random_audit_detection": detection}


def summary(rows):
    fragile = [row for row in rows if row["changed_bits"]]
    counts = {
        mode: {policy: np.asarray([row[mode][policy] for row in rows]).sum(axis=0).tolist()
               for policy in ("exact", "fractional", "lipschitz")}
        for mode in ("complete_certificates", "partial_certificates")
    }
    return {
        "profiles": len(rows), "certified": counts, "outcome_changes": len(fragile),
        "profitable": sum(row["profitable_bits"] > 0 for row in rows),
        "profitable_with_unique_winners": sum(row["unique_profitable_bits"] > 0 for row in rows),
        "expected_20_probe_miss_among_fragile": (1 - fmean(row["random_audit_detection"] for row in fragile)) if fragile else None,
    }


def run(directory):
    table = RobustTable(6, 3)
    witness = pair_separation(table)
    panels = {
        name: [trace(table, synthetic(seed, name), seed) for seed in range(start, start + size)]
        for name, start, size in (("clustered", 4201270, 128), ("independent", 4202270, 64))
    }
    for index, (name, expected) in enumerate(SOURCES):
        archive = (directory / name).read_bytes()
        assert sha256(archive).hexdigest() == expected
        _, projects, approvals = split_voting_data(archive)
        if approvals:
            start = 4203270 + 100 * index
            panels[name] = [trace(table, historical(seed, projects, approvals), seed) for seed in range(start, start + 20)]
    assert len(panels) == 7
    primary_panel = np.array([bool(row["changed_bits"]) for row in panels["clustered"]])
    bootstrap = np.random.default_rng(4208850).integers(0, len(primary_panel), (4000, len(primary_panel)))
    interval = np.quantile(primary_panel[bootstrap].mean(axis=1), [.025, .975]).tolist()
    return {"protocol_sha256": sha256(Path(__file__).with_name("FULL_DISCLOSURE_PROTOCOL.md").read_bytes()).hexdigest(),
            "primary": {"one_bit_fragility": float(primary_panel.mean()), "interval": interval,
                        "practical_frequency_reached": interval[0] >= .25},
            "exact_pair_separation": witness,
            "panels": {name: {"summary": summary(rows), "records": rows} for name, rows in panels.items()}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, required=True)
    output = cli.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--verify", type=Path)
    args = cli.parse_args()
    result = run(args.data_dir)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        assert serialized == args.verify.read_text(), "Full-disclosure replay differs"
    print(json.dumps({"primary": result["primary"], "exact_pair_separation": result["exact_pair_separation"],
                      "summaries": {name: item["summary"] for name, item in result["panels"].items()}}, indent=2))


if __name__ == "__main__":
    main()
