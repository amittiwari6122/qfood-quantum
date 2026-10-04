def decode(bits, candidates):
    """Map a bitstring back to (donation, request) assignments.
    Slack bits (beyond len(candidates)) are ignored."""
    out = []
    for c in candidates:
        if c.index < len(bits) and bits[c.index] == 1:
            out.append(c)
    return out


def to_payload(c, run_id=None):
    return {
        "donationId": c.donation_id, "requestId": c.request_id, "ngoId": c.ngo_id,
        "quantity": c.quantity, "distanceKm": c.distance_km,
        "deliveryMinutes": c.travel_minutes, "foodQuality": round(c.quality, 3),
        "freshness": round(c.freshness, 3), "urgency": round(c.urgency, 3),
        "remainingMinutes": c.remaining_minutes, "netBenefit": round(c.weight, 4),
    }
