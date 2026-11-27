# Stage 33 directional material transfer diagnostic
The previous measured contour shell is reconstructed from the verified report.
High-poly source UVs sample the explicitly supplied external albedo, normal and
roughness maps. Six depth-tested orthographic projections bake independent
albedo, projected tangent-space normal and ORM (AO=1, metal=0) textures.
The independently generated shell has new UVs and embedded glTF PBR textures.
Geometry and UV roundtrip are verified. Two matched views use the same direct
GGX approximation and source maps. Projected coverage is sampled inside faces.
Low-roughness projected components and repeated floor strips are hypotheses;
only strips with similar silhouette masks are shuffled together across maps.
This is a diagnostic transfer, not production per-building atlases, certified
original PBR, semantic window classification or arbitrary GIS reconstruction.
