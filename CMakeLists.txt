cmake_minimum_required(VERSION 3.12)
project(BSPOT)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BSPOT_ENABLE_CUDA "Build BSP-OT with optional CUDA acceleration" OFF)
option(BSPOT_USE_GEOMETRY_CENTRAL "Fetch geometry-central (needed for some apps)" ON)

if(BSPOT_ENABLE_CUDA)
  add_compile_definitions(BSPOT_ENABLE_CUDA=1)
endif()


include(CPM)
include(openmp)
include(eigen)
include(polyscope)
if(BSPOT_USE_GEOMETRY_CENTRAL)
  include(geometry-central)
endif()
include(spdlog)
include(spectra)

file(GLOB COMMON
    "common/*"
)

file(GLOB SOURCE
    "src/*.cpp"
    "src/*.h"
)
list(FILTER SOURCE EXCLUDE REGEX "cuda_backend_stub.cpp")

file(GLOB MESH
        "src_meshes/*"
)

file(GLOB IMAGE
        "src_images/*.hpp"
        "src_images/*.h"
        "src_images/*.cpp"
)

if(BSPOT_ENABLE_CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  enable_language(CUDA)
  add_library(bspot_cuda src/cuda_backend.cu)
  target_compile_definitions(bspot_cuda PUBLIC BSPOT_ENABLE_CUDA=1)
else()
  add_library(bspot_cuda src/cuda_backend_stub.cpp)
endif()
target_include_directories(bspot_cuda PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(bspot_cuda PUBLIC Eigen3::Eigen spdlog::spdlog ${CMAKE_DL_LIBS})

add_executable(bijections apps/bijections.cpp ${SOURCE})
target_link_libraries(bijections polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX bspot_cuda)

if(BSPOT_USE_GEOMETRY_CENTRAL)
  add_executable(manifold_sampling apps/manifold_sampling.cpp ${SOURCE} ${MESH})
  target_link_libraries(manifold_sampling polyscope spdlog geometry-central Eigen3::Eigen OpenMP::OpenMP_CXX Spectra bspot_cuda)
endif()

add_executable(persistance_diagrams_matching apps/persistance_diagrams_matching.cpp ${SOURCE})
target_link_libraries(persistance_diagrams_matching polyscope spdlog  Eigen3::Eigen OpenMP::OpenMP_CXX bspot_cuda)

add_executable(barycenters apps/barycenters.cpp ${SOURCE} ${IMAGE})
target_link_libraries(barycenters polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX Spectra bspot_cuda)

if(BSPOT_USE_GEOMETRY_CENTRAL)
  add_executable(general_coupling apps/general_coupling.cpp ${SOURCE} ${MESH})
  target_link_libraries(general_coupling polyscope spdlog Eigen3::Eigen geometry-central OpenMP::OpenMP_CXX Spectra bspot_cuda)
endif()

add_executable(color_transfer apps/color_transfer.cpp ${SOURCE} ${IMAGE})
target_link_libraries(color_transfer polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX bspot_cuda)

add_executable(stippling apps/stippling.cpp ${SOURCE} ${IMAGE})
target_link_libraries(stippling polyscope spdlog Eigen3::Eigen  OpenMP::OpenMP_CXX bspot_cuda)

add_executable(partial_transport apps/partial_transport.cpp ${SOURCE})
target_link_libraries(partial_transport polyscope spdlog  Eigen3::Eigen OpenMP::OpenMP_CXX bspot_cuda)

add_executable(scale_rigid_registration apps/scale_rigid_registration.cpp ${SOURCE})
target_link_libraries(scale_rigid_registration polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX bspot_cuda)

add_executable(advection_splatting apps/advection_splatting.cpp ${SOURCE} ${IMAGE})
target_link_libraries(advection_splatting polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX bspot_cuda)

add_executable(multi_marginals apps/multi_marginals.cpp ${SOURCE} ${IMAGE})
target_link_libraries(multi_marginals polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX Spectra bspot_cuda)

add_executable(viz_playground apps/viz_playground.cpp ${SOURCE})
target_link_libraries(viz_playground polyscope spdlog Eigen3::Eigen OpenMP::OpenMP_CXX bspot_cuda)
