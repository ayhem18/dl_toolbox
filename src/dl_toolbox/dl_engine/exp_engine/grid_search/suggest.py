"""Suggested values for a grid search axis. Not recipe JSON."""

from dl_toolbox.base.parsing.parse_numerical import parse_positive_int


def suggest_log_learning_rates(
    *, start_lr: float, end_lr: float, num: int
) -> tuple[float, ...]:
    if start_lr <= 0.0:
        raise ValueError(f"start_lr must be > 0, got {start_lr}")
    if end_lr <= 0.0:
        raise ValueError(f"end_lr must be > 0, got {end_lr}")
    if end_lr == start_lr:
        raise ValueError(f"end_lr must differ from start_lr, got {end_lr}")
    if end_lr < start_lr:
        raise ValueError(f"end_lr must be > start_lr, got {end_lr} <= {start_lr}")
    checked_num = parse_positive_int(num, label="num")
    if checked_num < 2:
        raise ValueError(f"num must be >= 2, got {checked_num}")
    ratio = end_lr / start_lr
    denom = float(checked_num - 1)
    rates: list[float] = [0.0] * checked_num
    for i in range(checked_num):
        rates[i] = start_lr * (ratio ** (float(i) / denom))
    return tuple(rates)


def suggest_lin_values(
    *, start: float, end: float, num: int
) -> tuple[float, ...]:
    if start == end:
        raise ValueError(f"end must differ from start, got {end}")
    checked_num = parse_positive_int(num, label="num")
    if checked_num < 2:
        raise ValueError(f"num must be >= 2, got {checked_num}")
    denom = float(checked_num - 1)
    values: list[float] = [0.0] * checked_num
    span = end - start
    for i in range(checked_num):
        values[i] = start + span * (float(i) / denom)
    return tuple(values)
