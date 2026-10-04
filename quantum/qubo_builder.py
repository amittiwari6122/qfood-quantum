"""Builds the QUBO  E(x) = x^T Q x + offset  (to be MINIMISED).

Objective  (per candidate k):
    linear  Q[k,k] = -(benefit_k - cost_k)          # reward good matches
Penalties:
    at most one request per donation   P1 * x_k x_l
    no request over-capacity           P2 * x_k x_l
    deliveries <= available partners   P3 * (sum_k x_k + sum_b 2^b s_b - M)^2
"""
from __future__ import annotations

import numpy as np

from .constraints import (delivery_capacity_slack_bits, one_request_per_donation,
                          request_over_capacity)


class QUBO:
    def __init__(self, n):
        self.n = n
        self.Q = np.zeros((n, n))
        self.offset = 0.0
        self.labels: list[str] = []
        self.objective_terms: list[dict] = []
        self.penalty_terms: list[dict] = []
        self.num_constraints = 0

    def add_linear(self, i, v):
        self.Q[i, i] += v

    def add_quadratic(self, i, j, v):
        if i == j:
            self.Q[i, i] += v
            return
        a, b = (i, j) if i < j else (j, i)
        self.Q[a, b] += v

    def energy(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def to_dict(self):
        nz = [[int(i), int(j), round(float(self.Q[i, j]), 6)]
              for i, j in zip(*np.nonzero(self.Q))]
        return {"size": self.n, "labels": self.labels, "offset": self.offset,
                "entries": nz, "dense": np.round(self.Q, 4).tolist() if self.n <= 40 else None}


def build_qubo(candidates, requests, available_partners, constraints):
    n_x = len(candidates)
    max_w = max([abs(c.weight) for c in candidates] + [1.0])
    pf = float(constraints.get("penaltyFactor", 2.5))
    P = pf * max_w + 1.0

    max_deliveries = constraints.get("maxDeliveries")
    if max_deliveries is None:
        max_deliveries = sum(int(p.get("capacity", 1)) for p in available_partners)
    n_donations = len({c.donation_id for c in candidates})
    use_slack = 0 < max_deliveries < min(n_donations, n_x)
    n_slack = delivery_capacity_slack_bits(max_deliveries) if use_slack else 0

    q = QUBO(n_x + n_slack)
    for c in candidates:
        q.labels.append(f"x[{c.donation_id}->{c.request_id}]")
        q.add_linear(c.index, -c.weight)
        q.objective_terms.append({"var": c.index, "label": q.labels[-1],
                                  "benefit": {k: round(v, 4) for k, v in c.benefit_terms.items()},
                                  "cost": {k: round(v, 4) for k, v in c.cost_terms.items()},
                                  "net": round(c.weight, 4)})

    pairs, groups = one_request_per_donation(candidates)
    for i, j in pairs:
        q.add_quadratic(i, j, P)
    q.num_constraints += groups
    q.penalty_terms.append({"name": "one request per donation", "weight": P, "pairs": len(pairs)})

    cap_pairs = request_over_capacity(candidates, requests,
                                      float(constraints.get("capacityTolerance", 0.15)))
    for i, j in cap_pairs:
        q.add_quadratic(i, j, P)
    q.num_constraints += len({candidates[i].request_id for i, _ in cap_pairs})
    q.penalty_terms.append({"name": "request not over-filled", "weight": P, "pairs": len(cap_pairs)})

    if use_slack:
        # (sum x + sum 2^b s_b - M)^2 expanded into QUBO form
        coeffs = [(k, 1.0) for k in range(n_x)]
        for b in range(n_slack):
            q.labels.append(f"slack[{b}]")
            coeffs.append((n_x + b, float(2 ** b)))
        M = float(max_deliveries)
        for a in range(len(coeffs)):
            ia, ca = coeffs[a]
            q.add_linear(ia, P * (ca * ca - 2 * M * ca))
            for b in range(a + 1, len(coeffs)):
                ib, cb = coeffs[b]
                q.add_quadratic(ia, ib, 2 * P * ca * cb)
        q.offset += P * M * M
        q.num_constraints += 1
        q.penalty_terms.append({"name": f"deliveries <= {int(M)} partners (slack bits {n_slack})",
                                "weight": P, "pairs": None})

    return q, {"penalty": P, "slackBits": n_slack, "maxDeliveries": max_deliveries,
               "decisionVariables": n_x}
