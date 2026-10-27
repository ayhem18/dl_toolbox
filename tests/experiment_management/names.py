"""Random project and package names. Not collected by pytest."""

from __future__ import annotations

import random
import string

N_TRIALS = 20


def random_project_name(rng: random.Random) -> str:
    alphabet = string.ascii_lowercase + string.digits
    n = rng.randint(6, 14)
    return "proj_" + "".join(rng.choice(alphabet) for _ in range(n))


def random_package_name(rng: random.Random) -> str:
    alphabet = string.ascii_lowercase
    n = rng.randint(4, 10)
    return "pkg_" + "".join(rng.choice(alphabet) for _ in range(n))
