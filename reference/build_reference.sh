#!/usr/bin/env bash
# Rebuild the C++ SPRITE v1.0 reference and regenerate the golden data used by
# sprite_py/tests. The original sources in ../sprite_c are never modified: a
# copy is patched for a modern macOS toolchain (Apple clang + Homebrew libomp,
# cfitsio, gsl). None of the patches touches numerical code.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
B=$HERE/sprite_c_build
OMP=$(brew --prefix libomp)

rm -rf "$B"
rsync -a --exclude .DS_Store ../sprite_c/ "$B/"

# 1. Toolchain: C++98 dialect, OpenMP through Homebrew libomp, recent CMake.
sed -i '' 's/cmake_minimum_required(VERSION 2.6)/cmake_minimum_required(VERSION 3.5)/' "$B/CMakeLists.txt"
sed -i '' "s|-DMACOS -DNO_DISP_IO -g0 -O2 -ffast-math -fomit-frame-pointer -fopenmp|-DMACOS -DNO_DISP_IO -std=gnu++98 -Wno-c++11-narrowing -w -Xpreprocessor -fopenmp -I$OMP/include -g0 -O2 -ffast-math -fomit-frame-pointer|" "$B/CMakeLists.txt"
sed -i '' "s|target_link_libraries(\${program}  sparse2d  sparse1d tools  \${GSL_LIBRARIES})|target_link_libraries(\${program}  sparse2d  sparse1d tools  \${GSL_LIBRARIES} -L$OMP/lib -lomp)|" "$B/CMakeLists.txt"
# 2. std::complex real()/imag() are not assignable in modern libc++ (complex FITS reader, unused by SPRITE).
sed -i '' 's/Mat(i,j).real() = Data(i,j,0);/Mat(i,j).real(Data(i,j,0));/; s/Mat(i,j).imag() = Data(i,j,1);/Mat(i,j).imag(Data(i,j,1));/' "$B/src/libtools/Arr_Fits.cc"
# 3. Global 'transform' clashes with std::transform in run_sprite.cc.
perl -pi -e 's/\btransform\b(?!_)/sp_transform/g unless /#include/' "$B/src/run_sprite.cc"

mkdir -p "$B/build" && cd "$B/build"
cmake .. > cmake.log
make -j8 run_sprite > make.log
make install > /dev/null
cd "$HERE"

# Harness exporting intermediate results (links the unmodified libraries).
clang++ -std=gnu++98 -w -DMACOS -DNO_DISP_IO -Xpreprocessor -fopenmp -O2 -ffast-math \
  -I"$OMP/include" -I"$B/src/libtools" -I"$B/src/libsparse1d" -I"$B/src/libsparse2d" \
  $(pkg-config --cflags cfitsio gsl) dump_ref.cc -o dump_ref \
  "$B/build/libsparse2d.a" "$B/build/libsparse1d.a" "$B/build/libtools.a" \
  $(pkg-config --libs cfitsio gsl) -L"$OMP/lib" -lomp

mkdir -p golden
cp ../sprite_c/test/cube_test.fits golden/
cd golden
for t in 2 24; do
  mkdir -p t$t units_t$t
  # Paper settings (section 6), plus -W diagnostics and -w residuals.
  ../sprite_c_build/bin/run_sprite -v -W -t $t -s 4 -r 2 -F -N -w resi.fits cube_test.fits solution.fits t$t/ > t$t/run.log
  ../dump_ref cube_test.fits units_t$t/ $t > /dev/null
done
echo "golden data regenerated in $HERE/golden"
