"""Unmodified oriented box primitive from OSM Building Grammar, Apache-2.0.
Upstream p-schulz/osm_building_grammar@81f1b50eae048645b2e620056a7cb2c84709bcfa.
Upstream addon attribution: FZI Forschungszentrum Informatik, 2026.
License: lab/vendor/osm_building_grammar/LICENSE.
PCG calls with a right-facing outward axis and validates the resulting winding.
"""
Point2D = tuple[float,float]
Point3D = tuple[float,float,float]

def _oriented_box(
    center: Point2D,
    tangent: Point2D,
    normal: Point2D,
    width: float,
    depth: float,
    height: float,
    bottom: float,
) -> tuple[list[Point3D], list[list[int]]]:
    half_width = width / 2.0
    half_depth = depth / 2.0
    corners = []
    for lateral, outward in [
        (-half_width, -half_depth),
        (half_width, -half_depth),
        (half_width, half_depth),
        (-half_width, half_depth),
    ]:
        corners.append((
            center[0] + tangent[0] * lateral + normal[0] * outward,
            center[1] + tangent[1] * lateral + normal[1] * outward,
        ))
    vertices = [(x, y, bottom) for x, y in corners] + [(x, y, bottom + height) for x, y in corners]
    faces = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ]
    return vertices, faces
