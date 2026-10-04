"""QAOA solver for a QUBO, built directly on Qiskit circuits.

1. QUBO  ->  Ising Hamiltonian  (x = (1 - z) / 2)
2. QAOA ansatz with p layers:  H^n  [ e^{-i gamma H_C}  e^{-i beta H_M} ]^p
3. Classical outer loop (COBYLA) minimises <psi(gamma,beta)|H_C|psi(gamma,beta)>,
   evaluated exactly from the Aer statevector.
4. The optimised circuit is sampled with shots (Aer locally, or an IBM
   Quantum backend in cloud mode). Sampled bitstrings are ranked by QUBO
   energy and handed to the validator - the best *valid* one wins.
"""
from __future__ import annotations

import os
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


def qubo_to_ising(Q: np.ndarray, offset: float):
    n = Q.shape[0]
    h = np.zeros(n)
    J = {}
    const = offset
    for i in range(n):
        h[i] -= Q[i, i] / 2
        const += Q[i, i] / 2
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                J[(i, j)] = Q[i, j] / 4
                h[i] -= Q[i, j] / 4
                h[j] -= Q[i, j] / 4
                const += Q[i, j] / 4
    return h, J, const


def energies_all_states(Q: np.ndarray, offset: float) -> np.ndarray:
    """QUBO energy of every computational basis state (index bit q = qubit q)."""
    n = Q.shape[0]
    idx = np.arange(2 ** n, dtype=np.uint32)
    bits = [((idx >> q) & 1).astype(np.float64) for q in range(n)]
    E = np.full(2 ** n, offset, dtype=np.float64)
    for i, j in zip(*np.nonzero(Q)):
        E += Q[i, j] * (bits[i] if i == j else bits[i] * bits[j])
    return E


def build_qaoa_circuit(h, J, p, measure=False):
    n = len(h)
    gammas = ParameterVector("γ", p)
    betas = ParameterVector("β", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for i in range(n):
            if h[i] != 0:
                qc.rz(2 * gammas[layer] * h[i], i)
        for (i, j), w in J.items():
            qc.rzz(2 * gammas[layer] * w, i, j)
        for i in range(n):
            qc.rx(2 * betas[layer], i)
    if measure:
        qc.measure_all()
    return qc, gammas, betas


class QAOASolver:
    def __init__(self, reps=2, restarts=3, maxiter=150, shots=4096, seed=7):
        self.reps = reps
        self.restarts = restarts
        self.maxiter = maxiter
        self.shots = shots
        self.seed = seed

    def solve(self, qubo, execution_mode="simulator", backend_name=None):
        t0 = time.time()
        Q, offset = qubo.Q, qubo.offset
        n = Q.shape[0]
        scale = float(np.max(np.abs(Q))) or 1.0
        Qn, offn = Q / scale, offset / scale
        h, J, _ = qubo_to_ising(Qn, offn)
        diag = energies_all_states(Qn, offn)

        sv_sim = AerSimulator(method="statevector", seed_simulator=self.seed)
        qc, gammas, betas = build_qaoa_circuit(h, J, self.reps)
        qc_sv = qc.copy()
        qc_sv.save_statevector()
        qc_sv = transpile(qc_sv, sv_sim)

        history = []

        def expectation(theta):
            bind = {gammas[k]: theta[k] for k in range(self.reps)}
            bind.update({betas[k]: theta[self.reps + k] for k in range(self.reps)})
            bound = qc_sv.assign_parameters(bind)
            sv = np.asarray(sv_sim.run(bound).result().get_statevector())
            val = float(np.dot(np.abs(sv) ** 2, diag))
            history.append(val * scale)
            return val

        rng = np.random.default_rng(self.seed)
        best = None
        for r in range(self.restarts):
            x0 = np.concatenate([rng.uniform(0, np.pi, self.reps),
                                 rng.uniform(0, np.pi / 2, self.reps)])
            res = minimize(expectation, x0, method="COBYLA",
                           options={"maxiter": self.maxiter, "rhobeg": 0.4})
            if best is None or res.fun < best.fun:
                best = res
        theta = best.x

        # ---- sampling / readout -------------------------------------------
        qc_m, g2, b2 = build_qaoa_circuit(h, J, self.reps, measure=True)
        bind = {g2[k]: theta[k] for k in range(self.reps)}
        bind.update({b2[k]: theta[self.reps + k] for k in range(self.reps)})
        bound = qc_m.assign_parameters(bind)

        backend_used = "aer_simulator"
        mode_used = "simulator"
        if execution_mode == "cloud":
            from .optimizer import run_on_ibm  # lazy import
            counts, backend_used = run_on_ibm(bound, self.shots, backend_name)
            mode_used = "cloud"
        else:
            sim = AerSimulator(seed_simulator=self.seed)
            counts = sim.run(transpile(bound, sim), shots=self.shots).result().get_counts()

        samples = []
        for bitstr, c in counts.items():
            bits = [int(b) for b in bitstr.replace(" ", "")[::-1]]  # qiskit is little-endian
            samples.append({"bits": bits, "count": c, "probability": c / self.shots,
                            "energy": qubo.energy(bits)})
        samples.sort(key=lambda s: s["energy"])

        return {
            "theta": {"gamma": theta[:self.reps].tolist(), "beta": theta[self.reps:].tolist()},
            "expectedEnergy": float(best.fun) * scale + 0.0,
            "history": history,
            "samples": samples,
            "numQubits": n,
            "reps": self.reps,
            "circuitDepth": qc.depth(),
            "gateCounts": dict(qc.count_ops()),
            "backend": backend_used,
            "executionMode": mode_used,
            "evaluations": len(history),
            "time": time.time() - t0,
        }
