// UNCOMPILED SDK 2.22 smoke-test candidate; NOT production adapter.
// Add to a locally generated UNIGINE C++ double-precision project, call from
// AppWorldLogic::init on the main thread, retain returned pointers until shutdown.
// Provide two writable engine data paths and a verified material file path.
#include <UnigineMesh.h>
#include <UnigineObjects.h>
#include <UnigineMathLib.h>
#include <stdexcept>

namespace pcg_sdk_probe {
using namespace Unigine;
using namespace Unigine::Math;

struct Scene {
    ObjectMeshStaticPtr unique;
    ObjectMeshClusterPtr cluster;
};

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

Scene build(const char *panel_path, const char *module_path, const char *material_path) {
    // Explicit split vertices, local meters, Z-up. Two CCW triangles in XY.
    auto mesh = Mesh::create();
    const int s = mesh->addSurface("roof");
    const vec3 positions[] = {{0,0,0},{4,0,0},{4,3,0},{0,3,0}};
    const vec2 uvs[] = {{0,0},{4,0},{4,3},{0,3}};
    for (int i=0; i<4; ++i) {
        mesh->addVertex(positions[i],s);
        mesh->addNormal(vec3(0,0,1),s);
        mesh->addTexCoord0(uvs[i],s);
    }
    for (int i : {0,1,2,0,2,3}) mesh->addIndex(i,s);
    require(mesh->createTangents(s)!=0,"panel tangent generation failed");
    mesh->createBounds();
    require(mesh->save(panel_path)!=0,"native panel write failed");

    auto module = Mesh::create();
    module->addBoxSurface("module",vec3(1.0f,0.2f,1.4f));
    require(module->save(module_path)!=0,"native module write failed");

    auto unique = ObjectMeshStatic::create(panel_path);
    require(bool(unique),"static node creation failed");
    unique->setMaterialFilePath(material_path,0);
    unique->setWorldTransform(translate(Vec3(0,0,2)));
    unique->setCollision(false,0);
    unique->setIntersection(true,0);

    auto cluster = ObjectMeshCluster::create(module_path);
    require(bool(cluster),"cluster node creation failed");
    cluster->setMaterialFilePath(material_path,0);
    cluster->setWorldTransform(translate(Vec3(0,0,0)));
    for (int i=0; i<16; ++i) {
        int id=cluster->addMeshTransform();
        cluster->setMeshTransform(id,translate(vec3(float(i%4)*2,8+float(i/4)*2,1)));
        // Optional second test only after declaring pcg_variant in surface layout
        // and compiling a Current Surface Parameters material:
        // cluster->setInstanceCustomParameterUInt(id,0,"pcg_variant",unsigned(i%4));
    }
    return {unique,cluster};
}
} // namespace pcg_sdk_probe
