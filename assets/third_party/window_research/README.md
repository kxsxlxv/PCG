# Window and interior mapping research assets (Stage 21)

Unmodified upstream textures, isolated from runtime assets. Both upstream projects publish MIT license notices (included verbatim in each directory). Retain notices for redistribution. No production UNIGINE integration or visual acceptance is implied by vendoring. Upstream asset provenance beyond the repositories' published license statements has not been independently audited.

## Sources
- Gaxil/Unity-InteriorMapping, MIT, commit 78194254efe646aee5c9784dc23338a263c14343. Source directory Assets/Textures. https://github.com/Gaxil/Unity-InteriorMapping
- codedgar/three-fenestra, MIT, commit d63feeb1d43fc0156d622eeb9f3e115532147113. Source directory starter. https://github.com/codedgar/three-fenestra

## Sampling contracts
- Gaxil basecolorFull: 4 vertically stacked blind-position variants; alpha participates in window compositing. Normal and RMW repeat per window tile; R = roughness, G = metallic, B = window-related mask (upstream uses B to select normal and blend reflection/occlusion; do not assume conventional ORM or invert without inspection). Base color sRGB; RMW and normal linear data. Decoration atlas is optional.
- three-fenestra rooms: 4x4 atlas of pre-lit perspective room images, not a cubemap or five independently baked room faces. Use compatible 2D interior mapping and inset/clamp each cell; choose apartment-appropriate variants. Overlay: 4x4 RGBA window dressings (alpha opacity), contains frames as well as curtains/blinds; avoid double mullions and avoid repeating foreground window frames behind independently authored ones. Color textures sRGB. Stable room IDs and matching day/night variant policy needed.
- Neither source supplies a verified metric window height map for POM. Generate geometry-derived depth/normal/AO for local sill/frame/reveal profiles, not from luminance of the photos. Far LOD should avoid full POM.
- The larger Gaxil BuildingAtlas.png (3,788,169 bytes) and DecorationAtlas2.png were not imported due to the transfer limit. They are optional relative to the three-fenestra starter atlas. Fetch from the pinned upstream commit if required.

## Additional acquisition
- PanelkaPack by Kureca (CC0, 173 MB): https://kureca.itch.io/panelka-pack — high priority for authentic Eastern European windows/doors/panel walls; not yet obtained. Select individual textures rather than commit the whole ZIP. Preserve CC0 provenance.
- ambientCG Facade019B / Facade020C (CC0): https://ambientcg.com/view?id=Facade019B and https://ambientcg.com/view?id=Facade020C — candidate facade material inputs, not verified as independent window atlases.
- wParallax free samples: https://wparallax.com/ — reference only; free download does not establish redistribution permission. Do not vendor without explicit applicable license.
- UNIGINE 2.22 Material Graph has Interior Mapping nodes; SDK agent should check built-in sample and actual texture layout before adapting.

## Acquisition follow-up
The wParallax license explicitly distinguishes the Apache-2.0 JIWindowbox script from its proprietary EXR textures, including free samples. See https://license-agreement.wparallax.com/. The original Autodesk-distributed Apache-2.0 OSL reference and its license are now included under `jiwindowbox/`, pinned to 4697d212a02e0732764793cb3cec581a321b33a7. The OSL code is a reference for cross-layout room intersection and layered foreground/midground composition, not a UNIGINE-compatible runtime shader. Keep commercial EXR maps outside Git and public CI artifacts unless explicit redistribution permission is obtained. A local licensed evaluation can be optional, not a required open-source dependency.

PanelkaPack's author explicitly declares CC0 on the official itch.io page; archive still pending. Import selected original window, door, facade and normal/PBR files with source attribution and hashes once acquired. The chat environment cannot resolve external download hosts and the itch.io page requires an interactive download; do not claim the archive has been imported. No generator changes or new rendering results in this acquisition commit.
