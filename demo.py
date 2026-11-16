"""Demonstrate the exact stopping certificate and a count-information separation."""
import json

import numpy as np

from voting import SearchSpace, Elicitation, audit


def demo():
    voters, candidates, seats = 22, 7, 3
    details = audit()["strict_witness"]
    space = SearchSpace(candidates, seats)
    full = np.ones(voters, dtype=int)
    reference = int(space.score[full].sum(axis=0).argmax())
    assert space.committees[reference] == tuple(range(seats))

    count_free = Elicitation(space, full, False)
    witness_count = 0
    for voter in range(voters):
        for candidate in range(seats, candidates):
            counterfactual = full.copy()
            counterfactual[voter] |= 1 << candidate
            assert int(space.score[counterfactual].sum(axis=0).argmax()) != reference
            assert not count_free.exact_certificate(count_free.candidate())
            count_free.query(voter, candidate)
            witness_count += 1
    assert count_free.exact_certificate(reference)

    informed = Elicitation(space, full, True)
    for voter in range(voters):
        if informed.unresolved(voter, 0):
            informed.query(voter, 0)
    assert informed.exact_certificate(reference)
    return {
        "coupled_certificate_checks": details,
        "unanimous_profile": {
            "voters": voters,
            "candidates": candidates,
            "seats": seats,
            "tie_broken_winner": list(space.committees[reference]),
            "count_free_lower_bound_in_binary_answers": witness_count,
            "count_free_construction_in_binary_answers": count_free.count,
            "count_first_construction_in_binary_answers": informed.count,
            "count_first_construction_in_numeric_answers": voters,
            "count_first_construction_total_responses": informed.count + voters,
        },
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
