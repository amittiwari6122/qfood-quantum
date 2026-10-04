"""Classical validation of a decoded quantum solution. The quantum result is
never trusted blindly: every assignment is re-checked against hard rules."""
from collections import defaultdict
from datetime import datetime, timezone

from .feature_builder import parse_time


def validate(assignments, donations, requests, available_partners, constraints, max_deliveries):
    errors = []
    now = parse_time(constraints.get("now")) or datetime.now(timezone.utc)
    dmap = {d["id"]: d for d in donations}
    rmap = {r["id"]: r for r in requests}
    tol = float(constraints.get("capacityTolerance", 0.15))

    seen = set()
    load = defaultdict(float)
    for a in assignments:
        d, r = dmap.get(a.donation_id), rmap.get(a.request_id)
        if d is None or r is None:
            errors.append(f"unknown entity in {a.donation_id}->{a.request_id}")
            continue
        if a.donation_id in seen:
            errors.append(f"duplicate assignment of donation {a.donation_id}")
        seen.add(a.donation_id)
        exp = parse_time(d.get("usableUntil"))
        if not exp or exp <= now:
            errors.append(f"donation {a.donation_id} expired")
        elif a.travel_minutes >= (exp - now).total_seconds() / 60:
            errors.append(f"donation {a.donation_id} expires before delivery")
        if constraints.get("requireVerified", True) and not r.get("ngoVerified"):
            errors.append(f"NGO for request {a.request_id} not verified")
        if d.get("aiStatus") == "UNSAFE":
            errors.append(f"donation {a.donation_id} flagged unsafe")
        if d.get("isVeg") is False and r.get("vegOnly"):
            errors.append(f"non-veg food to veg-only request {a.request_id}")
        load[a.request_id] += float(d.get("quantity") or 0)

    for rid, q in load.items():
        need = float(rmap[rid].get("quantity") or 1)
        n_assigned = sum(1 for a in assignments if a.request_id == rid)
        if n_assigned > 1 and q > need * (1 + tol):
            errors.append(f"request {rid} over-filled ({q:g} > {need:g})")

    if max_deliveries and len(assignments) > max_deliveries:
        errors.append(f"{len(assignments)} deliveries but only {max_deliveries} partner slots")
    if not available_partners and assignments:
        errors.append("no delivery partner available")
    return len(errors) == 0, errors
