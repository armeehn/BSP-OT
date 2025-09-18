project(BSPOT)
cmake_minimum_required(VERSION 3.12)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include(CPM)
include(openmp)
include(eigen)
include(polyscope)
include(geometry-central)
include(spdlog)
include(spectra)

file(GLOB COMMON
    "common/*"
)

file(GLOB SOURCE
    "src/*"
)

file(GLOB MESH
        "src_meshes/*"
)

file(GLOB IMAGE
        "src_images/*.hpp"
        "src_images/*.h"
        "src_images/*.cpp"
)

add_executable(bijections apps/bijections.cpp ${SOURCE})
target_link_libraries(bijections polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX)

add_executable(manifold_sampling apps/manifold_sampling.cpp ${SOURCE} ${MESH})
target_link_libraries(manifold_sampling polyscope spdlog geometry-central Eigen3::Eigen OpenMP::OpenMP_CXX Spectra)

add_executable(persistance_diagrams_matching apps/persistance_diagrams_matching.cpp ${SOURCE})
target_link_libraries(persistance_diagrams_matching polyscope spdlog  Eigen3::Eigen OpenMP::OpenMP_CXX)

add_executable(barycenters apps/barycenters.cpp ${SOURCE} ${IMAGE})
target_link_libraries(barycenters polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX)

add_executable(general_coupling apps/general_coupling.cpp ${SOURCE} ${MESH})
target_link_libraries(general_coupling polyscope spdlog Eigen3::Eigen geometry-central OpenMP::OpenMP_CXX)

add_executable(color_transfer apps/color_transfer.cpp ${SOURCE} ${IMAGE})
target_link_libraries(color_transfer polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX)

add_executable(stippling apps/stippling.cpp ${SOURCE} ${IMAGE})
target_link_libraries(stippling polyscope spdlog Eigen3::Eigen  OpenMP::OpenMP_CXX)

add_executable(partial_transport apps/partial_transport.cpp ${SOURCE})
target_link_libraries(partial_transport polyscope spdlog  Eigen3::Eigen OpenMP::OpenMP_CXX)

add_executable(scale_rigid_registration apps/scale_rigid_registration.cpp ${SOURCE})
target_link_libraries(scale_rigid_registration polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX)

add_executable(advection_splatting apps/advection_splatting.cpp ${SOURCE} ${IMAGE})
target_link_libraries(advection_splatting polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX)

add_executable(multi_marginals apps/multi_marginals.cpp ${SOURCE} ${IMAGE})
target_link_libraries(multi_marginals polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX)

