## Official source code of the "BSP-OT: Sparse transport plans between discrete measures in loglinear time" paper (SIGGRAPH Asia 2025). 
Baptiste Genest, Nicolas Bonneel, Vincent Nivoliers, David Coeurjolly.

![teaser](https://github.com/baptiste-genest/BSP-OT/blob/main/teaser.jpg)

# BSP-OT Compilation Guide

This guide explains how to compile the BSPOT project using CMake, focusing on the top-level `CMakeLists.txt`. It also lists all dependencies and how to obtain them.

## Requirements

- **CMake** >= 3.12
- **C++20** compatible compiler (e.g., GCC 10+, Clang 10+, MSVC 2019+)
- **git** (for fetching some dependencies)
- **Internet connection** (for fetching dependencies using CPM)

### Dependencies

The following libraries are required and are automatically handled by the CMake build system via CPM or included CMake scripts:

- [Eigen3](https://gitlab.com/libeigen/eigen) (version 3.4.0, downloaded automatically)
- [OpenMP](https://www.openmp.org/) (for parallelization, usually provided by your compiler)
- [Polyscope](https://github.com/nmwsharp/polyscope)
- [geometry-central](https://github.com/nmwsharp/geometry-central)
- [spdlog](https://github.com/gabime/spdlog)
- [Spectra](https://github.com/yixuan/spectra)

All dependencies except Eigen3 are also included via CMake include scripts (see the `cmake/` directory).

## Step-by-Step Compilation

1. **Go in code folder**
   ```bash
   cd BSP-OT
   ```

2. **Create a build directory**
   ```bash
   mkdir build
   cd build
   ```

3. **Configure the project with CMake**
   ```bash
   cmake ..
   ```
   - This will download and configure all dependencies using CPM and the scripts in `cmake/`.

4. **Build the project**
   ```bash
   cmake --build .
   ```
   - This will build all executables defined in the main `CMakeLists.txt`.

## Available Executables

After compilation, the following programs will be built (if their sources are present):

- `bijections`
- `manifold_sampling`
- `persistance_diagrams_matching`
- `barycenters`
- `color_transfer`
- `stippling`
- `scale_rigid_registration`

Each corresponds to a source file in the `apps/` directory.

## Example

to reproduce the figure 8, you can execute
```bash
./bijections --mu_file ../data/point_clouds/armadillo.pts --nb_trees 64 --viz
```

the --viz parameter allows to see the results with polyscope.

## Static parameters

To optimize performances, the code has some static parameters:

-for bijective applications (bijections, barycenters, persistance_diagrams_matching,color_transfer) you can compile with floats to get a speed-up without changing the quality. The other applications must use doubles. this is set by the type *scalar* defined in common/types.h. Double by default.

Each main file in apps is compiled with a static dimension, if you want to try 2D examples, please set "static_dim = 2".


