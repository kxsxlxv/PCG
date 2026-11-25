"""Stable, synthetic night-window intents shared across facade LODs.

Reuses the SHA-256 domain seed policy from the original research 06 material
probe. No measured occupancy model, interior or physical light simulation.
"""
from dataclasses import dataclass
import hashlib
import json

# Linear RGB emissive factors; bounded shared variants rather than unique
# per-window materials. Probabilities are illustrative, not regional data.
EMISSION = {"off": (0., 0., 0.), "warm": (.82, .52, .24),
            "neutral": (.66, .69, .57)}
PROBABILITY = {"house": .35, "apartments": .40, "office": .20}


@dataclass(frozen=True)
class WindowIntent:
    opening_id: str
    area_m2: float
    variant: str

    @property
    def emissive_linear(self):
        return EMISSION[self.variant]


def plan_night(plan, seed=17):
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    p = PROBABILITY[plan.profile.id]
    intents = []
    for facade in plan.facades:
        for opening in facade.openings:
            if opening.role not in ("window", "balcony_door"):
                continue
            key = json.dumps([seed, plan.building_id, opening.id, "occupancy-v1"],
                             separators=(",", ":"), ensure_ascii=False)
            d = hashlib.sha256(key.encode("utf-8")).digest()
            u = int.from_bytes(d[:8], "big") / 2**64
            variant = "off" if u >= p else ("warm" if d[8] < 192 else "neutral")
            intents.append(WindowIntent(opening.id, opening.width * opening.height, variant))
    intents.sort(key=lambda i: i.opening_id)
    if len({i.opening_id for i in intents}) != len(intents):
        raise ValueError("duplicate opening identity")
    return tuple(intents)


def mean_emission(intents):
    """Area-weighted linear RGB over glazing; not facade-wide radiance."""
    total = sum(i.area_m2 for i in intents)
    if not total:
        return (0., 0., 0.)
    return tuple(sum(i.area_m2 * i.emissive_linear[k] for i in intents) / total
                 for k in range(3))
