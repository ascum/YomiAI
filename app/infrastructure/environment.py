"""
app/infrastructure/environment.py — Simulated reward environment.

Moved from src/environment.py. Logic unchanged.
"""
import numpy as np

_USER_W = None


def reset_user():
    global _USER_W
    _USER_W = None


def get_reward(item_id, item_vec):
    global _USER_W
    if _USER_W is None:
        _USER_W = np.random.randn(len(item_vec))
    p = 1 / (1 + np.exp(-_USER_W @ item_vec))
    return int(np.random.rand() < p)
