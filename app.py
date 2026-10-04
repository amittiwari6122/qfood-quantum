import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from quantum.optimizer import MAX_QUBITS, solve

app = FastAPI(title="Quantum Matching Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


class SolveRequest(BaseModel):
    runId: str | None = None
    donations: list[dict] = Field(default_factory=list)
    requests: list[dict] = Field(default_factory=list)
    deliveryPartners: list[dict] = Field(default_factory=list)
    constraints: dict = Field(default_factory=dict)


@app.get("/health")
def health():
    mode = os.getenv("QUANTUM_MODE", "simulator")
    return {"status": "ok", "algorithm": "QAOA", "executionMode": mode,
            "maxQubitsPerBatch": MAX_QUBITS,
            "cloudConfigured": bool(os.getenv("IBM_QUANTUM_TOKEN")),
            "backend": os.getenv("IBM_QUANTUM_BACKEND") if mode == "cloud" else "aer_simulator"}


@app.post("/quantum/solve")
def quantum_solve(req: SolveRequest):
    try:
        return solve(req.model_dump())
    except RuntimeError as e:  # e.g. cloud mode not configured
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
