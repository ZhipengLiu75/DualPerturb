from __future__ import annotations

def time_split(total_days: int, time_length: int) -> tuple[range, range, range]:
    train_end = int(total_days * 0.70)
    valid_end = int(total_days * 0.85)
    if time_length <= 0:
        raise ValueError("time_length must be greater than 0")
    if time_length >= train_end:
        raise ValueError("time_length must be shorter than the training split")
    return range(time_length, train_end), range(train_end, valid_end), range(valid_end, total_days)
