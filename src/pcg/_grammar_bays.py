"""Unmodified window_offsets from OSM Building Grammar, Apache-2.0.
Upstream p-schulz/osm_building_grammar@81f1b50eae048645b2e620056a7cb2c84709bcfa.
See lab/vendor/osm_building_grammar/LICENSE and UPSTREAM.md.
Upstream addon attribution: FZI Forschungszentrum Informatik, 2026.
Only this pure function is adopted into the production namespace.
"""
def window_offsets(length: float, width: float, spacing: float, margin: float) -> list[float]:
    usable = length - margin * 2.0
    if usable < width:
        return []
    count = max(1, int((usable + spacing - width) // spacing))
    total_width = (count - 1) * spacing + width
    start = (length - total_width) / 2.0 + width / 2.0
    return [start + index * spacing for index in range(count)]
