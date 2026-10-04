"""Orchestrates feature building -> QUBO -> QAOA -> decode -> validate.

Problems larger than MAX_QUBITS are decomposed into sequential sub-problems
(batches of donations). Each batch is a genuine QUBO solved by QAOA; capacity
used by earlier batches is carried into later ones. This is reported in the
response as `decomposition` so the UI never hides it.
"""
from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict

import numpy as np

from .feature_builder import build_candidates
from .qaoa_solver import QAOASolver, energies_all_states
from .qubo_builder import build_qubo
from .solution_decoder import decode, to_payload
from .validator import validate

MAX_QUBITS = int(os.getenv("QUANTUM_MAX_QUBITS", "16"))


def run_on_ibm(circuit, shots, backend_name=None):
    """Execute the optimised QAOA circuit on IBM Quantum hardware via
    qiskit-ibm-runtime. Raises if not configured - never silently falls back."""
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        raise RuntimeError("QUANTUM_MODE=cloud but IBM_QUANTUM_TOKEN is not set")
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as e:
        raise RuntimeError("Install qiskit-ibm-runtime to use cloud mode") from e
    from qiskit import transpile
    service = QiskitRuntimeService(channel=os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform"),
                                   token=token, instance=os.getenv("IBM_QUANTUM_INSTANCE") or None)
    backend = service.backend(backend_name) if backend_name else service.least_busy(operational=True, simulator=False)
    isa = transpile(circuit, backend=backend, optimization_level=3)
    job = SamplerV2(mode=backend).run([isa], shots=shots)
    counts = job.result()[0].data.meas.get_counts()
    return counts, backend.name


def _trim_top_k(candidates, k):
    by_d = defaultdict(list)
    for c in candidates:
        by_d[c.donation_id].append(c)
    kept, dropped = [], []
    for cs in by_d.values():
        cs.sort(key=lambda c: c.weight, reverse=True)
        kept += cs[:k]
        dropped += cs[k:]
    for i, c in enumerate(kept):
        c.index = i
    return kept, dropped


def _reindex(cands):
    for i, c in enumerate(cands):
        c.index = i
    return cands


def _solve_batch(cands, requests, partners, constraints, mode, backend, solver):
    cands = [c for c in cands if c.weight > -1e9]
    _reindex(cands)
    qubo, meta = build_qubo(cands, requests, partners, constraints)
    if qubo.n == 0:
        return None
    if qubo.n > MAX_QUBITS:
        raise ValueError(f"batch needs {qubo.n} qubits > limit {MAX_QUBITS}")
    res = solver.solve(qubo, mode, backend)

    chosen, chosen_errors, rejected = None, None, 0
    for s in res["samples"]:
        assigned = decode(s["bits"], cands)
        ok, errs = validate(assigned, constraints["_donations"], requests, partners,
                            constraints, meta["maxDeliveries"])
        if ok:
            chosen = (s, assigned)
            break
        rejected += 1
        chosen_errors = errs

    exact = None
    if constraints.get("verifyWithExactSolver", True) and qubo.n <= 20:
        E = energies_all_states(qubo.Q, qubo.offset)
        exact = float(E.min())

    return {"qubo": qubo, "meta": meta, "result": res, "chosen": chosen,
            "rejectedSamples": rejected, "lastErrors": chosen_errors,
            "exactMinEnergy": exact, "candidates": cands}


def solve(payload: dict) -> dict:
    t0 = time.time()
    donations = payload.get("donations") or []
    requests = payload.get("requests") or []
    partners = payload.get("deliveryPartners") or []
    constraints = dict(payload.get("constraints") or {})
    constraints["_donations"] = donations
    mode = (constraints.get("executionMode") or os.getenv("QUANTUM_MODE", "simulator")).lower()
    backend_name = os.getenv("IBM_QUANTUM_BACKEND") or None
    solver = QAOASolver(reps=int(constraints.get("reps", os.getenv("QAOA_REPS", 2))),
                        restarts=int(constraints.get("restarts", 3)),
                        maxiter=int(constraints.get("maxIter", 150)),
                        shots=int(constraints.get("shots", 4096)))
    run_id = payload.get("runId") or str(uuid.uuid4())

    candidates, pruned, avail = build_candidates(donations, requests, partners, constraints)
    top_k = int(constraints.get("topKPerDonation", 3))
    candidates, dropped = _trim_top_k(candidates, top_k)
    for c in dropped:
        pruned.append({"donationId": c.donation_id, "requestId": c.request_id,
                       "reason": f"outside top-{top_k} candidates for this donation"})

    base = {"runId": run_id, "algorithm": "QAOA", "executionMode": mode,
            "numberOfDonations": len(donations), "numberOfRequests": len(requests),
            "possibleAssignments": len(donations) * len(requests), "pruned": pruned}

    if not candidates:
        return {**base, "status": "no_feasible", "valid": False, "assignments": [],
                "message": "No feasible quantum matching found.", "objectiveValue": None,
                "numberOfVariables": 0, "numberOfConstraints": 0,
                "executionTime": round(time.time() - t0, 3), "backend": None, "batches": []}

    # ---- decompose into batches of donations that fit in MAX_QUBITS -------
    by_d = defaultdict(list)
    for c in candidates:
        by_d[c.donation_id].append(c)
    order = sorted(by_d, key=lambda d: max(c.weight for c in by_d[d]), reverse=True)
    batches, cur = [], []
    slack_budget = 3
    for d in order:
        if cur and sum(len(by_d[x]) for x in cur) + len(by_d[d]) + slack_budget > MAX_QUBITS:
            batches.append(cur)
            cur = []
        cur.append(d)
    if cur:
        batches.append(cur)

    remaining_need = {r["id"]: float(r.get("quantity") or 1) for r in requests}
    partner_slots = constraints.get("maxDeliveries")
    if partner_slots is None:
        partner_slots = sum(int(p.get("capacity", 1)) for p in avail)

    final, batch_reports, total_obj = [], [], 0.0
    first_qubo, backend_used = None, None
    total_vars = total_cons = 0
    for bi, ds in enumerate(batches):
        if partner_slots <= 0:
            break
        reqs_now = [{**r, "quantity": remaining_need[r["id"]]} for r in requests
                    if remaining_need[r["id"]] > 0]
        live = {r["id"] for r in reqs_now}
        cands = [c for d in ds for c in by_d[d] if c.request_id in live]
        if not cands:
            continue
        sub_constraints = {**constraints, "maxDeliveries": partner_slots}
        out = _solve_batch(cands, reqs_now, avail, sub_constraints, mode, backend_name, solver)
        if out is None:
            continue
        r = out["result"]
        backend_used = r["backend"]
        total_vars += out["qubo"].n
        total_cons += out["qubo"].num_constraints
        if first_qubo is None:
            first_qubo = out
        rep = {"batch": bi, "qubits": out["qubo"].n, "decisionVariables": out["meta"]["decisionVariables"],
               "slackBits": out["meta"]["slackBits"], "evaluations": r["evaluations"],
               "expectedEnergy": round(r["expectedEnergy"], 4),
               "exactMinEnergy": None if out["exactMinEnergy"] is None else round(out["exactMinEnergy"], 4),
               "rejectedSamples": out["rejectedSamples"], "circuitDepth": r["circuitDepth"]}
        if out["chosen"]:
            s, assigned = out["chosen"]
            rep["selectedEnergy"] = round(s["energy"], 4)
            rep["selectedProbability"] = round(s["probability"], 4)
            if out["exactMinEnergy"] is not None:
                rep["reachedExactOptimum"] = abs(s["energy"] - out["exactMinEnergy"]) < 1e-6
            total_obj += s["energy"]
            for a in assigned:
                final.append(a)
                remaining_need[a.request_id] -= a.quantity
                partner_slots -= 1
        else:
            rep["validationErrors"] = out["lastErrors"]
        batch_reports.append(rep)

    fq = first_qubo
    viz = None
    if fq:
        r = fq["result"]
        viz = {"qubo": fq["qubo"].to_dict(),
               "objectiveTerms": fq["qubo"].objective_terms,
               "penaltyTerms": fq["qubo"].penalty_terms,
               "history": [round(v, 4) for v in r["history"]],
               "distribution": [{"bits": "".join(map(str, s["bits"])), "probability": round(s["probability"], 4),
                                 "energy": round(s["energy"], 4)}
                                for s in sorted(r["samples"], key=lambda s: -s["probability"])[:16]],
               "theta": r["theta"], "reps": r["reps"], "gateCounts": r["gateCounts"],
               "circuitDepth": r["circuitDepth"]}

    valid = len(final) > 0
    return {**base,
            "status": "success" if valid else "no_feasible",
            "message": None if valid else "No feasible quantum matching found.",
            "executionMode": "cloud" if backend_used and backend_used != "aer_simulator" else "simulator",
            "backend": backend_used,
            "valid": valid,
            "assignments": [to_payload(a, run_id) for a in final],
            "objectiveValue": round(total_obj, 4) if valid else None,
            "numberOfVariables": total_vars,
            "numberOfConstraints": total_cons,
            "executionTime": round(time.time() - t0, 3),
            "decomposition": {"maxQubitsPerBatch": MAX_QUBITS, "batches": len(batch_reports),
                              "note": "Single QUBO" if len(batch_reports) <= 1 else
                              "Problem split into sequential QUBO batches to fit the simulator qubit limit"},
            "batches": batch_reports,
            "visualization": viz}
