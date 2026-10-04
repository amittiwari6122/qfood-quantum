"""Constraint definitions expressed as QUBO penalty terms.

All constraints are encoded as quadratic penalties so the whole problem is
a single Quadratic Unconstrained Binary Optimisation.
"""
from __future__ import annotations

import math
from collections import defaultdict


def one_request_per_donation(candidates):
    """A donation can go to at most one request.
    For each pair of variables sharing a donation: P * x_k * x_l.
    (At-most-one needs no linear term.)"""
    groups = defaultdict(list)
    for c in candidates:
        groups[c.donation_id].append(c.index)
    pairs = []
    for idxs in groups.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pairs.append((idxs[a], idxs[b]))
    return pairs, len(groups)


def request_over_capacity(candidates, requests, tolerance=0.15):
    """Two donations sent to the same request whose combined quantity
    exceeds the need (plus tolerance) are penalised as a pair."""
    need = {r["id"]: float(r.get("quantity") or 1) for r in requests}
    groups = defaultdict(list)
    for c in candidates:
        groups[c.request_id].append(c)
    pairs = []
    for rid, cs in groups.items():
        cap = need.get(rid, 1) * (1 + tolerance)
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                if cs[a].quantity + cs[b].quantity > cap:
                    pairs.append((cs[a].index, cs[b].index))
    return pairs


def delivery_capacity_slack_bits(max_assignments: int) -> int:
    """Number of binary slack bits needed to encode sum(x) + s = M, 0<=s<=M."""
    if max_assignments <= 0:
        return 1
    return int(math.floor(math.log2(max_assignments))) + 1
