#include <UnigineConsole.h>
#include <UnigineCamera.h>
#include <UnigineEngine.h>
#include <UnigineGame.h>
#include <UnigineImage.h>
#include <UnigineImport.h>
#include <UnigineInit.h>
#include <UnigineInput.h>
#include <UnigineLights.h>
#include <UnigineLog.h>
#include <UnigineLogic.h>
#include <UnigineMaterials.h>
#include <UnigineMesh.h>
#include <UnigineNodes.h>
#include <UnigineObjects.h>
#include <UniginePlayers.h>
#include <UnigineRender.h>
#include <UnigineTextures.h>
#include <UnigineViewport.h>
#include <UnigineVisualizer.h>
#include <UnigineWindowManager.h>
#include <UnigineWorld.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
using namespace Unigine;
using namespace Unigine::Math;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

std::string setting(const char *name, const char *fallback)
{
    const char *value = std::getenv(name);
    return value && *value ? value : fallback;
}

void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

struct Geometry
{
    int triangles = 0;
    int surfaces = 0;
    Vec3 minimum{1e30};
    Vec3 maximum{-1e30};
};

void check_bounds(const Geometry &actual, const Geometry &expected)
{
    constexpr double tolerance_m = 0.002;
    require(length(actual.minimum - expected.minimum) <= tolerance_m &&
            length(actual.maximum - expected.maximum) <= tolerance_m,
        "Imported bounds differ from canonical Z-up bounds");
}

MaterialPtr tint(const vec4 &color)
{
    auto base = Materials::findManualMaterial("Unigine::mesh_base");
    require(bool(base), "Unigine::mesh_base unavailable");
    auto material = base->inherit();
    require(bool(material) && material->findParameter("albedo_color") >= 0,
        "mesh_base has no albedo_color");
    material->setParameterFloat4("albedo_color", color);
    return material;
}

void persist_material(const fs::path &data, const std::string &path, MaterialPtr &material)
{
    if (setting("PCG_LAB_SAVE_SCENE", "").empty())
        return;
    if (fs::exists(data / path))
    {
        auto existing = Materials::loadMaterial(path.c_str());
        require(bool(existing), "Could not load the saved preview material");
        material = existing;
        return;
    }
    fs::create_directories((data / path).parent_path());
    require(material->createMaterialFile(path.c_str()), "Could not save preview material");
}

// Adapted from Create Mesh Primitives and research/09_integration/sdk_smoke_candidate.cpp.
ObjectMeshStaticPtr native_fixture()
{
    const char *path = "native_probe/asymmetric.mesh";
    auto mesh = Mesh::create();
    const std::array<vec3, 4> points{{{0, 0, 0}, {4, 0, 0}, {0, 2.5f, 0}, {0, 0, 1.25f}}};
    const std::array<const char *, 2> names{{"roof", "wall"}};
    const std::array<vec3, 2> normals{{{0, 0, 1}, {0, 1, 0}}};
    const std::array<std::array<int, 3>, 2> triangles{{{{0, 1, 2}}, {{0, 3, 1}}}};
    const std::array<std::array<vec2, 3>, 2> uvs{{
        {{{0, 0}, {2, 0}, {0, 1.25f}}},
        {{{0.125f, 0.25f}, {0.125f, 1.5f}, {2.125f, 0.25f}}}
    }};
    for (int s = 0; s < 2; ++s)
    {
        mesh->addSurface(names[s]);
        for (int v = 0; v < 3; ++v)
        {
            mesh->addVertex(points[triangles[s][v]], s);
            mesh->addNormal(normals[s], s);
            mesh->addTexCoord0(uvs[s][v], s);
            mesh->addIndex(v, s);
        }
        require(mesh->createTangents(s), "Panel tangent generation failed");
    }
    mesh->createBounds();
    require(mesh->save(path), "Native MESH save failed");
    auto reloaded = Mesh::create();
    require(reloaded->load(path), "Native MESH load failed");
    require(reloaded->getNumSurfaces() == 2, "Expected two reloaded fixture surfaces");
    for (int s = 0; s < 2; ++s)
    {
        Log::message("PCG mesh surface %d: CV=%d TV=%d CI=%d TI=%d tangent=%d UV=%d\n", s,
            reloaded->getNumCVertex(s), reloaded->getNumTVertex(s), reloaded->getNumCIndices(s),
            reloaded->getNumTIndices(s), reloaded->getNumTangents(s), reloaded->getNumTexCoords0(s));
        require(reloaded->getNumCVertex(s) == 3 && reloaded->getNumTVertex(s) == 3 &&
                reloaded->getNumTangents(s) == 3 && reloaded->getNumTexCoords0(s) == 3,
            "Missing reloaded fixture vertex attributes");
        require(std::string(reloaded->getSurfaceName(s)) == names[s] &&
                reloaded->getNumCIndices(s) == 3 && reloaded->getNumTIndices(s) == 3,
            "Fixture surface roles or indices changed");
        for (int v = 0; v < 3; ++v)
        {
            require(length(reloaded->getVertex(v, s) - points[triangles[s][v]]) < 0.001f &&
                    length(reloaded->getTangent(v, s).getNormal() - normals[s]) < 0.001f &&
                    length(reloaded->getTexCoord0(v, s) - uvs[s][v]) < 0.001f &&
                    reloaded->getIndex(v, s) == v,
                "Fixture vertex, UV, hard normal, or winding changed");
        }
    }
    const auto box = reloaded->getBoundBox();
    check_bounds({2, 2, Vec3(box.minimum), Vec3(box.maximum)}, {2, 2, Vec3(0.0), Vec3(4, 2.5, 1.25)});
    auto node = ObjectMeshStatic::create(path);
    require(bool(node), "Native fixture node creation failed");
    node->setName("PCG native roof and wall");
    node->setMaterial(tint(vec4(0.72f, 0.29f, 0.11f, 1)), 0);
    node->setMaterial(tint(vec4(0.1f, 0.32f, 0.7f, 1)), 1);
    Log::message("PCG G1 MESH roundtrip: 2 surfaces, 2 triangles, 4 x 2.5 x 1.25 m\n");
    return node;
}

void inspect_import(const NodePtr &node, Geometry &geometry)
{
    if (node->getType() == Node::OBJECT_MESH_STATIC)
    {
        const auto object = checked_ptr_cast<ObjectMeshStatic>(node);
        const auto mesh = Mesh::create();
        require(mesh->load(object->getMeshPath()), "Could not reopen imported native mesh");
        const auto transform = node->getWorldTransform();
        for (int s = 0; s < mesh->getNumSurfaces(); ++s)
        {
            const int indices = mesh->getNumTIndices(s);
            require(indices > 0 && indices % 3 == 0, "Unexpected imported triangle buffer");
            geometry.triangles += indices / 3;
            ++geometry.surfaces;
            for (int v = 0; v < mesh->getNumCVertex(s); ++v)
            {
                const Vec3 p = transform * Vec3(mesh->getVertex(v, s));
                geometry.minimum.x = std::min(geometry.minimum.x, p.x);
                geometry.minimum.y = std::min(geometry.minimum.y, p.y);
                geometry.minimum.z = std::min(geometry.minimum.z, p.z);
                geometry.maximum.x = std::max(geometry.maximum.x, p.x);
                geometry.maximum.y = std::max(geometry.maximum.y, p.y);
                geometry.maximum.z = std::max(geometry.maximum.z, p.z);
            }
            const auto material = object->getMaterial(s);
            require(bool(material), "Missing imported material");
            Log::message("PCG import surface %s [%d triangles]\n", mesh->getSurfaceName(s), indices / 3);
            for (int t = 0; t < material->getNumTextures(); ++t)
            {
                const char *path = material->getTexturePath(t);
                if (path && *path)
                    Log::message("  %s = %s\n", material->getTextureName(t), path);
            }
        }
    }
    for (int child = 0; child < node->getNumChildren(); ++child)
        inspect_import(node->getChild(child), geometry);
}

Geometry load_expected(const fs::path &input, const std::string &mode)
{
    std::ifstream file(input / "expected.txt");
    require(bool(file), "Missing preview/expected.txt. Generate the preview package first.");
    Geometry geometry;
    std::string name;
    while (file >> name >> geometry.triangles >> geometry.surfaces >>
        geometry.minimum.x >> geometry.minimum.y >> geometry.minimum.z >>
        geometry.maximum.x >> geometry.maximum.y >> geometry.maximum.z)
    {
        if (name == mode)
            return geometry;
    }
    throw std::runtime_error("Missing geometry expectations for " + mode);
}

struct MaterialBank
{
    std::map<std::string, MaterialPtr> materials;
    std::map<std::string, TexturePtr> albedo_textures;
};

NodePtr build_preview(const fs::path &data, const std::string &package, const std::string &mode,
    MaterialBank &bank)
{
    const auto input = data / package;
    const auto expected = load_expected(input, mode);
    std::ifstream source(input / (mode + ".pcg"));
    std::string version;
    int count;
    require(bool(source >> version >> count) && version == "PCG_PREVIEW_V1" &&
            count == expected.surfaces && count > 0 && count < 128,
        "Invalid native preview header");
    auto mesh = Mesh::create();
    std::vector<MaterialPtr> materials;
    Geometry actual;
    for (int surface = 0; surface < count; ++surface)
    {
        std::string name;
        int triangles;
        vec4 emission(0, 0, 0, 1);
        require(bool(source >> name >> triangles >> emission.x >> emission.y >> emission.z) &&
                name.find_first_not_of("abcdefghijklmnopqrstuvwxyz_0123456789") == std::string::npos &&
                triangles > 0 && triangles <= expected.triangles - actual.triangles,
            "Invalid native preview material group");
        mesh->addSurface(name.c_str());
        for (int index = 0; index < 3 * triangles; ++index)
        {
            vec3 point;
            vec3 normal;
            vec2 uv;
            require(bool(source >> point.x >> point.y >> point.z >> normal.x >> normal.y >> normal.z >> uv.x >> uv.y),
                "Truncated native preview geometry");
            mesh->addVertex(point, surface);
            mesh->addNormal(normal, surface);
            mesh->addTexCoord0(uv, surface);
            mesh->addIndex(index, surface);
        }
        require(mesh->createTangents(surface), "Could not create preview surface tangents");
        actual.triangles += triangles;
        auto &material = bank.materials[name];
        if (material)
        {
            materials.push_back(material);
            continue;
        }
        material = tint(vec4(1));
        const auto map_dir = package + "/maps/" + name.substr(0, name.find("__")) + "/";
        for (const auto &binding : std::array<std::pair<const char *, const char *>, 4>{{
                 {"albedo", "albedo.png"}, {"normal", "normal.png"}, {"shading", "shading.png"},
                 {"ambient_occlusion", "ao.png"}}})
        {
            const auto file = map_dir + binding.second;
            require(fs::exists(data / file), "Missing native preview texture");
            const int texture = material->findTexture(binding.first);
            require(texture >= 0, "Missing required mesh_base texture slot");
            material->setTexturePath(texture, file.c_str());
        }
        const int occlusion = material->findState("ao_map");
        require(occlusion >= 0, "mesh_base AO map state unavailable");
        material->setState(occlusion, 1);
        if (emission.x > 0 || emission.y > 0 || emission.z > 0)
        {
            require(material->findState("emission") >= 0 && material->findParameter("emission_color") >= 0,
                "mesh_base emission controls unavailable");
            material->setState("emission", 1);
            material->setParameterFloat4("emission_color", emission);
        }
        persist_material(data, "native_probe/" + package + "/materials_v3/" + name + ".mat", material);
        const int albedo = material->findTexture("albedo");
        const auto albedo_path = map_dir + "albedo.png";
        auto &color_texture = bank.albedo_textures[albedo_path];
        if (!color_texture)
        {
            color_texture = Texture::create();
            require(color_texture->load(albedo_path.c_str(),
                        material->getTextureSamplerFlags(albedo) | Texture::FORMAT_SRGB),
                "Could not load preview albedo with explicit sRGB");
            require(bool(color_texture->getFormatFlags() & Texture::FORMAT_SRGB),
                "Renderer did not retain the requested albedo sRGB flag");
        }
        material->setTexture(albedo, color_texture);
        if (bank.materials.size() == 1)
            Log::message("PCG albedo runtime binding: sRGB=%d\n",
                int(bool(color_texture->getFormatFlags() & Texture::FORMAT_SRGB)));
        materials.push_back(material);
    }
    std::string extra;
    require(!(source >> extra) && actual.triangles == expected.triangles, "Unexpected preview payload");
    mesh->createBounds();
    const auto bounds = mesh->getBoundBox();
    actual.minimum = Vec3(bounds.minimum);
    actual.maximum = Vec3(bounds.maximum);
    check_bounds(actual, expected);
    const auto file = "native_probe/" + package + "/" + mode + ".mesh";
    require(mesh->save(file.c_str()), "Could not save generated native preview mesh");
    auto object = ObjectMeshStatic::create(file.c_str());
    require(bool(object) && object->getNumSurfaces() == count, "UNIGINE could not load preview surfaces");
    object->setName(("PCG " + mode).c_str());
    for (int s = 0; s < count; ++s)
    {
        object->setMaterial(materials.at(s), s);
        object->setCollision(false, s);
        object->setIntersection(false, s);
    }
    Log::message("PCG native %s: %d triangles, %d materials, bounds %.5f %.5f %.5f to %.5f %.5f %.5f\n",
        mode.c_str(), actual.triangles, count,
        actual.minimum.x, actual.minimum.y, actual.minimum.z,
        actual.maximum.x, actual.maximum.y, actual.maximum.z);
    return object;
}

NodePtr import_preview(const fs::path &data, const std::string &package, const std::string &mode)
{
    const auto input = data / package;
    const auto output = input / ("import_" + mode);
    fs::create_directories(output);
    require(Import::isSupportedExtension("glb"), "GLB importer is unavailable");
    const auto source = package + "/" + mode + ".glb";
    Console::run("import_info");
    auto importer = Import::createImporterByFileName((input / (mode + ".glb")).generic_string().c_str());
    require(bool(importer), "Could not create GLB importer instance");
    require(importer->init(source.c_str()), "GLB importer could not initialize input");
    require(importer->import((output.generic_string() + "/").c_str()), "GLB importer could not convert input");
    const std::string file = importer->getOutputFilepath();
    Log::message("PCG importer output: %s\n", file.c_str());
    require(!file.empty(), "GLB importer did not produce a scene");
    auto node = World::loadNode(file.c_str(), false);
    require(bool(node), "Could not load imported scene node");
    node->setName(("PCG " + mode).c_str());

    Geometry actual;
    inspect_import(node, actual);
    Log::message("PCG import %s: %d triangles, %d surfaces, bounds %.5f %.5f %.5f to %.5f %.5f %.5f\n",
        mode.c_str(), actual.triangles, actual.surfaces,
        actual.minimum.x, actual.minimum.y, actual.minimum.z,
        actual.maximum.x, actual.maximum.y, actual.maximum.z);
    const auto expected = load_expected(input, mode);
    require(actual.triangles == expected.triangles && actual.surfaces == expected.surfaces,
        "Imported triangle or material surface count changed");
    check_bounds(actual, expected);
    return node;
}

struct View
{
    const char *name;
    Vec3 eye;
    Vec3 target;
    vec3 up{0, 0, 1};
};

const std::array<View, 4> views{{
    {"aerial", {56, -38, 48}, {26, 15, 4}},
    {"street", {8, -24, 2.0}, {22, 5, 4}},
    {"facade", {13, -11, 2.6}, {18.5, 0, 3.7}},
    {"roof", {26, 17, 57}, {26, 17, 0}, {0, 1, 0}},
}};
const View fixture_view{"fixture", {7, 5, 4.5}, {1.6, 1.1, 0.5}};
const std::array<View, 5> terrain_views{{
    {"aerial", {30, -24, 22}, {6, 4, 4}},
    {"street", {13, -16, 2}, {5, 0, 2.5}},
    {"facade", {4, -6, 2.2}, {7, 0, 3}},
    {"roof", {6, 4.5, 28}, {6, 4.5, 0}, {0, 1, 0}},
    {"contact", {-7, -2, 1.5}, {1, 4.5, 0.8}},
}};

ObjectLandscapeTerrainPtr selected_terrain()
{
    Vector<NodePtr> nodes;
    World::getNodesByType(Node::OBJECT_LANDSCAPE_TERRAIN, nodes);
    require(nodes.size() == 1, "Expected exactly one landscape terrain");
    auto terrain = checked_ptr_cast<ObjectLandscapeTerrain>(nodes[0]);
    require(bool(terrain) && terrain->isEnabled() && terrain->getNumSurfaces() == 1 &&
            terrain->getIntersection(0),
        "Landscape terrain height intersection is unavailable");
    return terrain;
}

// Only the selected landscape surface participates in this world-XY query.
std::optional<double> landscape_height(const ObjectLandscapeTerrainPtr &terrain, const Vec2 &xy)
{
    const auto bounds = terrain->getWorldBoundBox(0);
    if (!std::isfinite(bounds.minimum.z) || !std::isfinite(bounds.maximum.z) ||
        bounds.minimum.z > bounds.maximum.z)
        return std::nullopt;
    const auto inverse = terrain->getIWorldTransform();
    const Vec3 begin(xy.x, xy.y, bounds.maximum.z + 10.0);
    const Vec3 end(xy.x, xy.y, bounds.minimum.z - 10.0);
    auto hit = ObjectIntersection::create();
    if (!terrain->getIntersection(inverse * begin, inverse * end, hit, 0))
        return std::nullopt;
    const Vec3 point = terrain->getWorldTransform() * hit->getPoint();
    if (!std::isfinite(point.z) || std::hypot(point.x - xy.x, point.y - xy.y) > 0.02)
        return std::nullopt;
    return point.z;
}

void probe_landscape()
{
    const auto terrain = selected_terrain();
    const auto origin = terrain->getWorldPosition();
    Log::message("PCG landscape %s at %.5f %.5f %.5f, intersection precision %.3f\n",
        terrain->getName(), origin.x, origin.y, origin.z, terrain->getIntersectionPrecision());

    // Probe control points near the origin and one definitely outside the 8 km source layers.
    constexpr double tolerance_m = 0.15;
    const std::array<Vec2, 6> controls{{{0, 0}, {8, 3}, {13, 16}, {4, 19}, {6, 10}, {100000, 100000}}};
    int agreement = 0;
    for (const auto &xy : controls)
    {
        auto fetch = LandscapeFetch::create();
        fetch->setUsesHeight(true);
        const bool coverage = fetch->fetchForce(xy);
        const auto hit = landscape_height(terrain, xy);
        if (&xy == &controls.back())
            require(!hit && !coverage, "The outside landscape control unexpectedly returned a height");
        if (!hit || !coverage)
        {
            Log::message("PCG terrain %.3f %.3f coverage ray=%d fetch=%d\n",
                xy.x, xy.y, int(hit.has_value()), int(coverage));
            continue;
        }
        const auto fetched_z = static_cast<double>(fetch->getHeight());
        const auto difference = std::abs(*hit - fetched_z);
        Log::message("PCG terrain %.3f %.3f ray_z=%.6f fetch_z=%.6f diff=%.6f\n",
            xy.x, xy.y, *hit, fetched_z, difference);
        if (difference <= tolerance_m)
            ++agreement;
    }
    Log::message("PCG terrain diagnostic: %d sampled controls agree within %.2f m\n", agreement, tolerance_m);
    require(agreement == 5, "Landscape control heights did not all agree");

    const std::array<Vec2, 4> offsets{{{12, 0}, {12, 9}, {0, 9}, {5, 4}}};
    const std::array<double, 2> steps{{40.0, 120.0}};
    for (const auto step : steps)
    {
        int found = 0;
        for (int ring = 0; ring <= 24 && found < 8; ++ring)
            for (int y = -ring; y <= ring && found < 8; ++y)
                for (int x = -ring; x <= ring && found < 8; ++x)
                {
                    if (std::max(std::abs(x), std::abs(y)) != ring)
                        continue;
                    const Vec2 candidate(x * step, y * step);
                    const auto first = landscape_height(terrain, candidate);
                    if (!first)
                        continue;
                    double low = *first;
                    double high = *first;
                    bool valid = true;
                    for (const auto &offset : offsets)
                    {
                        const auto height = landscape_height(terrain, candidate + offset);
                        if (!height)
                        {
                            valid = false;
                            break;
                        }
                        low = std::min(low, *height);
                        high = std::max(high, *height);
                        if (high - low > 0.12)
                        {
                            valid = false;
                            break;
                        }
                    }
                    if (valid)
                    {
                        ++found;
                        Log::message("PCG patch candidate %.0f %.0f z %.5f corner relief %.5f m\n",
                            candidate.x, candidate.y, high, high - low);
                    }
                }
        if (found)
            break;
    }
}

struct TerrainPoint
{
    Vec3 position;
    bool accepted;
};

NodePtr place_package(const fs::path &data, const std::string &package, const std::string &mode,
    const Vec2 &origin, bool use_landscape, std::vector<TerrainPoint> &diagnostics, Vec3 &camera_offset,
    MaterialBank &bank)
{
    const auto terrain = use_landscape ? selected_terrain() : ObjectLandscapeTerrainPtr{};
    std::ifstream points(data / package / "placement.txt");
    require(bool(points), "Missing placement.txt in preview package");
    auto container = NodeDummy::create();
    container->setName("PCG buildings");
    constexpr double max_gap_m = 0.12;
    std::string id;
    int placed = 0;
    int total = 0;
    Vec2 anchor;
    int size;
    while (points >> id)
    {
        ++total;
        require(total <= 16 && bool(points >> anchor.x >> anchor.y >> size) &&
                size > 0 && size <= 100000 && id.find_first_not_of(
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_0123456789") == std::string::npos,
            "Invalid building placement manifest");
        const Vec2 xy = anchor + origin;
        int missing = 0;
        double low = 1e30;
        double high = -1e30;
        const auto start = diagnostics.size();
        for (int i = 0; i < size; ++i)
        {
            Vec2 local;
            require(bool(points >> local.x >> local.y), "Truncated placement points");
            if (!terrain)
                continue;
            const Vec2 sample = xy + local;
            const auto height = landscape_height(terrain, sample);
            if (!height)
            {
                ++missing;
                Log::message("PCG sample %s %.5f %.5f no_height\n", id.c_str(), sample.x, sample.y);
                continue;
            }
            low = std::min(low, *height);
            high = std::max(high, *height);
            diagnostics.push_back({Vec3(sample.x, sample.y, *height), false});
            Log::message("PCG sample %s %.5f %.5f %.5f\n", id.c_str(), sample.x, sample.y, *height);
        }
        if (missing)
        {
            Log::message("PCG placement %s review_required coverage_missing %d/%d\n", id.c_str(), missing, size);
            continue;
        }
        if (terrain && high - low > max_gap_m)
        {
            Log::message("PCG placement %s review_required relief %.5f m, min %.5f max %.5f, samples %d\n",
                id.c_str(), high - low, low, high, size);
            continue;
        }
        const double base = terrain ? high : 0.0;
        auto building = build_preview(data, package, mode + "-" + id, bank);
        building->setName(id.c_str());
        building->setParent(container);
        building->setWorldPosition(Vec3(xy.x, xy.y, base));
        for (std::size_t i = start; i < diagnostics.size(); ++i)
            diagnostics[i].accepted = true;
        if (!placed)
            camera_offset = Vec3(origin.x, origin.y, base);
        ++placed;
        Log::message("PCG placement %s accepted base_z %.5f, min %.5f max %.5f, samples %d\n",
            id.c_str(), base, terrain ? low : 0.0, base, size);
    }
    Log::message("PCG placement complete: %d/%d, max allowed sampled gap %.2f m\n", placed, total, max_gap_m);
    require(total > 0, "Preview package is empty");
    if (!placed && !diagnostics.empty())
        camera_offset = Vec3(origin.x, origin.y, diagnostics.front().position.z);
    return container;
}

class Preview final : public WorldLogic
{
public:
    int init() override
    {
        try
        {
            mode = setting("PCG_LAB_MODE", "day");
            if (mode == "terrain-probe")
            {
                probe_landscape();
                return 1;
            }
            require(mode == "native" || mode == "marker" || mode == "day" || mode == "night",
                "PCG_LAB_MODE must be native, marker, day, or night");
            const auto placement = setting("PCG_LAB_PLACEMENT", "flat_preview");
            landscape = placement == "landscape_local";
            require(landscape || placement == "flat_preview", "Unknown preview placement mode");
            require(!landscape || mode == "day" || mode == "night", "Landscape expects a building package");
            require(!landscape || setting("PCG_LAB_OFFICIAL_IMPORTER", "0") != "1",
                "The official importer diagnostic runs on the flat reference");
            package = setting("PCG_LAB_PACKAGE", "preview-native");
            require(package.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789_") ==
                    std::string::npos, "Invalid preview package name");
            origin = Vec2(std::stod(setting("PCG_LAB_ORIGIN_X", "0")), std::stod(setting("PCG_LAB_ORIGIN_Y", "0")));
            require(std::isfinite(origin.x) && std::isfinite(origin.y), "Invalid preview origin");
            const auto capture = setting("PCG_LAB_CAPTURE", "");
            if (!capture.empty())
            {
                capture_dir = fs::absolute(capture);
                fs::create_directories(capture_dir);
                capture_texture = Texture::create();
                require(capture_texture->create2D(1280, 720, Texture::FORMAT_RGBA8,
                            Texture::SAMPLER_FILTER_LINEAR | Texture::FORMAT_USAGE_RENDER),
                    "Could not allocate the screenshot render target");
            }
            data = Engine::get()->getDataPath();
            configure_layout();
            root = NodeDummy::create();
            root->setName("PCGPreview");
            if (!landscape)
            {
                auto ground_mesh = Mesh::create();
                const auto ground_width = std::max(160.0, span + 80.0);
                ground_mesh->addBoxSurface("flat_reference", vec3(float(ground_width), float(ground_width), 0.2f));
                const auto ground_path = "native_probe/" + package + "/flat_reference.mesh";
                require(ground_mesh->save(ground_path.c_str()), "Could not save the flat reference");
                auto ground = ObjectMeshStatic::create(ground_path.c_str());
                require(bool(ground), "Could not load the flat reference");
                ground->setName("Flat reference at Z=0");
                ground->setParent(root);
                ground->setWorldPosition(Vec3(origin.x + center.x, origin.y + center.y, -0.1));
                auto material = tint(vec4(0.34f, 0.36f, 0.35f, 1));
                persist_material(data, "native_probe/flat_reference.mat", material);
                ground->setMaterial(material, 0);

                auto sun = LightWorld::create(vec4(1, 0.94f, 0.83f, 1));
                sun->setParent(root);
                sun->setWorldDirection(normalize(vec3(0.6f, 0.4f, -0.85f)), vec3(0, 0, 1));
                sun->setIntensity(mode == "night" ? 0.03f : 1.25f);
            }
            else
            {
                Visualizer::setEnabled(true);
                if (mode == "night")
                {
                    const auto sun = checked_ptr_cast<LightWorld>(World::getNodeByName("sun"));
                    if (sun)
                        sun->setIntensity(0.03f);
                }
            }
            rebuild_content();
            camera = PlayerDummy::create();
            camera->setName("PCG Preview Camera");
            camera->setParent(root);
            camera->setFov(55);
            camera->setZNear(0.1f);
            camera->setZFar(landscape ? 6000.0f : 500.0f);
            Game::setPlayer(camera);
            Game::setEnabled(true);
            Render::setEnabled(true);
            if (!landscape || mode == "night")
            {
                Render::setEnvironment(false);
                Render::setBackgroundColor(mode == "night" ? vec4(0.02f, 0.04f, 0.09f, 1) :
                                                               vec4(0.30f, 0.43f, 0.58f, 1));
                Render::setCameraMode(Render::CAMERA_MODE_CLASSIC);
                Render::setExposureMode(Render::EXPOSURE_MODE_DISABLED);
                Render::setExposure(0.8f);
                Render::setTonemapper(true);
            }
            WindowManager::getMainWindow()->setCamera(camera->getCamera());
            select_view(0);
            const auto save = setting("PCG_LAB_SAVE_SCENE", "");
            if (!save.empty())
            {
                require(save.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789_") ==
                        std::string::npos, "Invalid generated world name");
                const auto file = save + ".world";
                require(!fs::exists(data / file), "Generated world already exists");
                require(World::saveWorld(file.c_str()), "Could not save the generated world");
                require(World::saveNode((save + ".node").c_str(), root), "Could not export the preview node tree");
                Log::message("PCG saved world %s\n", file.c_str());
            }
            Log::message("PCG ready: %s\n", mode.c_str());
        }
        catch (const std::exception &error)
        {
            Log::error("PCG FAILED: %s\n", error.what());
            failed = true;
        }
        return 1;
    }

    int update() override
    {
        if (failed || mode == "terrain-probe" || Input::isKeyDown(Input::KEY_ESC))
        {
            Engine::get()->quit();
            return 1;
        }
        if (capture_dir.empty() && setting("PCG_LAB_VALIDATE", "0") == "1")
        {
            if (Clock::now() - camera_changed < std::chrono::milliseconds(300))
                return 1;
            try
            {
                const int nodes = content->getNumChildren();
                rebuild_content();
                require(content->getNumChildren() == nodes, "Root child count changed after regeneration");
                Log::message("PCG regeneration: %d children before and after\n", nodes);
            }
            catch (const std::exception &error)
            {
                Log::error("PCG rebuild failed: %s\n", error.what());
                failed = true;
            }
            Engine::get()->quit();
            return 1;
        }
        if (capture_dir.empty())
        {
            const std::array<Input::KEY, 5> keys{{Input::KEY_F1, Input::KEY_F2, Input::KEY_F3, Input::KEY_F4, Input::KEY_F6}};
            for (int n = 0; n < view_count(); ++n)
                if (Input::isKeyDown(keys[n]))
                    select_view(n);
            if (Input::isKeyDown(Input::KEY_F5))
            {
                try
                {
                    rebuild_content();
                    select_view(view_index);
                }
                catch (const std::exception &error)
                {
                    Log::error("PCG rebuild failed: %s\n", error.what());
                    failed = true;
                }
            }
            draw_diagnostics();
            return 1;
        }
        draw_diagnostics();
        if (Clock::now() - camera_changed < std::chrono::seconds(2))
            return 1;
        if (!shot)
        {
            const auto path = (capture_dir / (std::string(current_view().name) + ".png")).generic_string();
            capture_viewport->renderTexture2D(capture_camera, capture_texture, 1280, 720, false);
            if (++capture_frames < 8)
                return 1;
            Render::asyncTransferTextureToImage(nullptr, MakeCallback([path](ImagePtr pixels)
            {
                if (!Render::isFlipped())
                    pixels->flipY();
                if (pixels->save(path.c_str()))
                    Log::message("PCG screenshot: %s\n", path.c_str());
                else
                    Log::error("PCG screenshot write failed: %s\n", path.c_str());
            }), capture_texture);
            camera_changed = Clock::now();
            shot = true;
        }
        else if (view_index + 1 == view_count())
        {
            Engine::get()->quit();
        }
        else
        {
            select_view(view_index + 1);
        }
        return 1;
    }

    int shutdown() override
    {
        content.clear();
        camera.clear();
        root.clear();
        capture_camera.clear();
        capture_viewport.clear();
        capture_texture.clear();
        material_bank.materials.clear();
        material_bank.albedo_textures.clear();
        return 1;
    }

    bool failed = false;

private:
    bool single_view() const { return mode == "native" || mode == "marker"; }
    int view_count() const { return single_view() ? 1 : landscape ? int(terrain_views.size()) : int(views.size()); }
    View current_view() const
    {
        if (single_view())
            return fixture_view;
        if (landscape && view_index == 4)
            return terrain_views[view_index];
        if (span < 6.0)
        {
            const Vec3 focal(center.x, center.y, 0.7);
            const char *name = views[view_index].name;
            if (view_index == 3)
                return {name, focal + Vec3(0.0, 0.0, span * 1.2), focal, {0, 1, 0}};
            return {name, focal + Vec3(span * 0.5, -span, span * 0.5), focal};
        }
        if (span > 65.0 && (view_index == 0 || view_index == 3))
        {
            const Vec3 focal(center.x, center.y, 3.0);
            if (view_index == 0)
                return {"aerial", focal + Vec3(span * 0.26, -span * 0.62, span * 0.52), focal};
            return {"roof", focal + Vec3(0.0, 0.0, span * 0.92), focal, {0, 1, 0}};
        }
        return span <= 17.0 ? terrain_views[view_index] : views[view_index];
    }

    void configure_layout()
    {
        if (single_view())
            return;
        std::ifstream input(data / package / "layout.txt");
        if (!input)
            return;
        Vec2 low;
        Vec2 high;
        require(bool(input >> low.x >> low.y >> high.x >> high.y), "Malformed preview layout");
        span = std::max(high.x - low.x, high.y - low.y);
        require(std::isfinite(span) && span > 0.0 && span < 5000.0, "Preview layout span is out of range");
        center = (low + high) * 0.5;
    }

    void rebuild_content()
    {
        if (content)
            content.deleteForce();
        diagnostics.clear();
        camera_offset = Vec3(origin.x, origin.y, 0);
        if (mode == "native")
            content = native_fixture();
        else if (setting("PCG_LAB_OFFICIAL_IMPORTER", "0") == "1")
            content = import_preview(data, package, mode);
        else if (mode == "marker")
            content = build_preview(data, package, mode, material_bank);
        else
            content = place_package(data, package, mode, origin, landscape, diagnostics, camera_offset, material_bank);
        content->setParent(root);
    }

    void draw_diagnostics() const
    {
        if (landscape && std::string(current_view().name) == "contact")
            for (const auto &point : diagnostics)
                Visualizer::renderPoint3D(point.position + Vec3(0, 0, 0.03), 0.006f,
                    point.accepted ? vec4_green : vec4_red, true, 0.0f, false);
    }

    void select_view(int index)
    {
        view_index = index;
        const auto &view = current_view();
        camera->setWorldPosition(view.eye + camera_offset);
        camera->setWorldDirection(normalize(vec3(view.target - view.eye)), view.up);
        if (!capture_dir.empty())
        {
            capture_viewport = Viewport::create();
            capture_viewport->setSkipFlags(capture_viewport->getSkipFlags() & ~Viewport::SKIP_VISUALIZER);
            capture_camera = Camera::create();
            capture_camera->setModelview(lookAt(view.eye + camera_offset, view.target + camera_offset, view.up));
            const float clip = landscape ? 6000.0f : float(std::max(500.0, span * 2));
            capture_camera->setProjection(perspective(55.0f, 1280.0f / 720.0f, 0.1f, clip));
            capture_frames = 0;
        }
        camera_changed = Clock::now();
        shot = false;
    }

    std::string mode;
    std::string package;
    fs::path data;
    Vec2 origin{0.0, 0.0};
    Vec2 center{26.0, 17.0};
    double span = 52.0;
    Vec3 camera_offset{0.0};
    bool landscape = false;
    std::vector<TerrainPoint> diagnostics;
    MaterialBank material_bank;
    fs::path capture_dir;
    NodeDummyPtr root;
    NodePtr content;
    PlayerDummyPtr camera;
    ViewportPtr capture_viewport;
    CameraPtr capture_camera;
    TexturePtr capture_texture;
    int capture_frames = 0;
    Clock::time_point camera_changed;
    int view_index = 0;
    bool shot = false;
};
} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t **argv)
#else
int main(int argc, char **argv)
#endif
{
    Preview preview;
    Unigine::EnginePtr engine(argc, argv);
    engine->main(nullptr, &preview);
    return preview.failed ? 1 : 0;
}
