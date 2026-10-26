"""Shared generators for data_contract tests. Not collected by pytest."""

from __future__ import annotations

import random
import string

N_TRIALS = 40


def random_generating_process_name(rng: random.Random) -> str:
    alphabet = string.ascii_lowercase + string.digits
    n = rng.randint(6, 18)
    return "run_" + "".join(rng.choice(alphabet) for _ in range(n))


def shuffled_mapping(rng: random.Random, payload: dict) -> dict:
    items = list(payload.items())
    rng.shuffle(items)
    return dict(items)


def random_json_object(rng: random.Random) -> dict:
    n_keys = rng.randint(1, 6)
    payload: dict = {}
    for i in range(n_keys):
        key = f"k{i}_{rng.randint(0, 10_000)}"
        kind = rng.choice(("int", "str", "bool", "nested"))
        if kind == "int":
            payload[key] = rng.randint(-10_000, 10_000)
        elif kind == "str":
            payload[key] = "".join(
                rng.choice(string.ascii_letters) for _ in range(rng.randint(1, 12))
            )
        elif kind == "bool":
            payload[key] = rng.choice((True, False))
        else:
            payload[key] = {"n": rng.randint(0, 50), "flag": rng.choice((True, False))}
    return payload
