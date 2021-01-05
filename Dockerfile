FROM dolfinx/dev-env

ARG DOLFINX_BUILD_TYPE=Release
ARG DOLFINX_CMAKE_CXX_FLAGS="-Ofast -march=znver2"
ENV PETSC_ARCH=linux-gnu-real-32

RUN apt-get update && \
    apt-get install -y texlive-latex-extra texlive-science texlive-fonts-recommended dvipng vim && \
    apt-get install -y clang

RUN pip3 install git+https://github.com/FEniCS/fiat.git --upgrade && \
    pip3 install git+https://github.com/FEniCS/ufl.git --upgrade && \
    pip3 install git+https://github.com/FEniCS/ffcx.git --upgrade && \
    pip3 install git+https://github.com/michalhabera/dolfiny.git@michal/fix-bs --upgrade && \
    pip3 install h5py lxml meshio && \
    rm -rf /usr/local/include/dolfin /usr/local/include/dolfin.h

RUN git clone --branch master https://github.com/FEniCS/dolfinx.git && \
    cd dolfinx && \
    mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=${DOLFINX_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${DOLFINX_CMAKE_CXX_FLAGS} ../cpp/ && \
    make -j 4 install && \
    cd /

RUN cd dolfinx/python && \
    CXXFLAGS=${DOLFINX_CMAKE_CXX_FLAGS} pip3 -v install .

RUN mkdir -p ~/.config/dolfinx && \
    echo '{"cffi_extra_compile_args": ["-Ofast", "-march=znver2"], "cffi_libraries": ["m"]}' >> ~/.config/dolfinx/dolfinx_jit_parameters.json
