"""Turns raw donations / requests / delivery partners into candidate
assignment pairs with normalised features.

Every candidate pair (donation i, request j) becomes one binary variable
x(i,j) in the QUBO. Pairs that are *hard* infeasible (expired food,
unverified NGO, food would expire before it can arrive, ...) are pruned
here so they never consume a qubit. Their reasons are reported back so
the UI can show why a pair was not considered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

EARTH_RADIUS_KM = 6371.0


def parse_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass
class Candidate:
    index: int
    donation_id: str
    request_id: str
    ngo_id: str
    quantity: float
    distance_km: float
    travel_minutes: float
    quality: float          # 0..1
    freshness: float        # 0..1 share of usable window still remaining
    urgency: float          # 0..1 how soon the NGO needs food
    compatibility: float    # 0..1 category / diet match
    quantity_fit: float     # 0..1 how well quantity matches need
    beneficiary_impact: float  # 0..1
    expiry_risk: float      # 0..1 how tight arrival is vs expiry
    remaining_minutes: float
    benefit_terms: dict = field(default_factory=dict)
    cost_terms: dict = field(default_factory=dict)
    weight: float = 0.0     # net benefit (higher = better)


DEFAULT_WEIGHTS = {
    "quality": 3.0,
    "freshness": 2.0,
    "urgency": 2.0,
    "beneficiary": 1.5,
    "compatibility": 2.5,
    "distance": 2.0,
    "travel": 1.0,
    "expiry_risk": 3.0,
    "quantity_mismatch": 1.5,
}


def build_candidates(donations, requests, partners, constraints):
    now = parse_time(constraints.get("now")) or datetime.now(timezone.utc)
    weights = {**DEFAULT_WEIGHTS, **(constraints.get("weights") or {})}
    max_distance = float(constraints.get("maxDistanceKm", 25))
    speed_kmh = float(constraints.get("avgSpeedKmh", 20))
    handling_min = float(constraints.get("handlingMinutes", 15))
    min_quality = float(constraints.get("minQualityScore", 0.5))
    require_verified = bool(constraints.get("requireVerified", True))
    min_partners = int(constraints.get("minAvailablePartners", 0))

    available_partners = [p for p in partners if p.get("available", True)]
    excluded = set(constraints.get("excludedPairs") or [])  # "donationId|requestId" rejected earlier
    candidates: list[Candidate] = []
    pruned: list[dict] = []
    max_beneficiaries = max([r.get("beneficiaries", 1) or 1 for r in requests] + [1])

    for d in donations:
        d_exp = parse_time(d.get("usableUntil"))
        d_prep = parse_time(d.get("preparedAt")) or now
        d_pick_end = parse_time(d.get("pickupEnd"))
        quality = d.get("qualityScore")
        quality = 0.6 if quality is None else float(quality)
        if quality > 1:
            quality /= 100.0

        for r in requests:
            reason = None
            if f'{d["id"]}|{r["id"]}' in excluded:
                reason = "previously rejected by this NGO"
            elif d_exp is None:
                reason = "donation has no usable-until time"
            elif d_exp <= now:
                reason = "donation expired"
            elif d.get("status") in {"EXPIRED", "CANCELLED", "COMPLETED"}:
                reason = f"donation status {d.get('status')}"
            elif d.get("aiStatus") == "UNSAFE":
                reason = "AI flagged high-risk visual condition"
            elif quality < min_quality:
                reason = "quality score below threshold"
            elif require_verified and not r.get("ngoVerified", False):
                reason = "NGO not verified"
            elif require_verified and d.get("donorVerified") is False:
                reason = "donor not verified"
            elif d.get("isVeg") is False and r.get("vegOnly"):
                reason = "request is veg-only"
            elif len(available_partners) <= min_partners - 1:
                reason = "no delivery partner available"

            if reason:
                pruned.append({"donationId": d["id"], "requestId": r["id"], "reason": reason})
                continue

            dist = haversine_km(d["lat"], d["lng"], r["lat"], r["lng"])
            if dist > max_distance:
                pruned.append({"donationId": d["id"], "requestId": r["id"],
                               "reason": f"distance {dist:.1f} km > {max_distance} km"})
                continue

            travel = dist / speed_kmh * 60 + handling_min
            remaining = (d_exp - now).total_seconds() / 60
            if travel >= remaining:
                pruned.append({"donationId": d["id"], "requestId": r["id"],
                               "reason": "food would expire before arrival"})
                continue
            if d_pick_end and d_pick_end <= now:
                pruned.append({"donationId": d["id"], "requestId": r["id"],
                               "reason": "pickup window closed"})
                continue
            r_before = parse_time(r.get("requiredBefore"))
            if r_before and r_before <= now + _minutes(travel):
                pruned.append({"donationId": d["id"], "requestId": r["id"],
                               "reason": "cannot arrive before NGO deadline"})
                continue

            window = max((d_exp - d_prep).total_seconds() / 60, remaining, 1)
            freshness = max(0.0, min(1.0, remaining / window))
            expiry_risk = max(0.0, min(1.0, travel / remaining))

            if r_before:
                hours_left = (r_before - now).total_seconds() / 3600
                urgency = 1.0 / (1.0 + max(hours_left, 0) / 6.0)
            else:
                urgency = 0.3
            if r.get("priority") == "HIGH":
                urgency = min(1.0, urgency + 0.25)
            elif r.get("priority") == "CRITICAL":
                urgency = min(1.0, urgency + 0.45)

            cat_d = (d.get("category") or "").lower()
            cats_r = [c.lower() for c in (r.get("categories") or [])]
            compat = 1.0 if (not cats_r or cat_d in cats_r or "any" in cats_r) else 0.25

            need = float(r.get("quantity") or 1)
            qty = float(d.get("quantity") or 0)
            fit = min(qty, need) / max(qty, need) if max(qty, need) > 0 else 0
            beneficiaries = (r.get("beneficiaries") or 1) / max_beneficiaries

            c = Candidate(
                index=len(candidates), donation_id=d["id"], request_id=r["id"],
                ngo_id=r.get("ngoId", r["id"]), quantity=min(qty, need),
                distance_km=round(dist, 3), travel_minutes=round(travel, 1),
                quality=quality, freshness=freshness, urgency=urgency,
                compatibility=compat, quantity_fit=fit,
                beneficiary_impact=beneficiaries, expiry_risk=expiry_risk,
                remaining_minutes=round(remaining, 1),
            )
            c.benefit_terms = {
                "quality": weights["quality"] * quality,
                "freshness": weights["freshness"] * freshness,
                "urgency": weights["urgency"] * urgency,
                "beneficiary": weights["beneficiary"] * beneficiaries,
                "compatibility": weights["compatibility"] * compat,
            }
            c.cost_terms = {
                "distance": weights["distance"] * min(dist / max_distance, 1.0),
                "travel": weights["travel"] * min(travel / 120.0, 1.0),
                "expiry_risk": weights["expiry_risk"] * expiry_risk,
                "quantity_mismatch": weights["quantity_mismatch"] * (1 - fit),
            }
            c.weight = sum(c.benefit_terms.values()) - sum(c.cost_terms.values())
            candidates.append(c)

    return candidates, pruned, available_partners


def _minutes(m):
    from datetime import timedelta
    return timedelta(minutes=m)
