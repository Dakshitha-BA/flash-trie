# Copyright (c) 2026 Microsoft Corporation
# Licensed under the MIT License. See COPYING.md.

"""Build the flashtrie CUDA extension with the repository CMake project.

``pip install .`` is the supported way to import the library. The version
written into the package is the ``VERSION`` file.
"""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        cmake = shutil.which("cmake")
        nvcc = shutil.which("nvcc")
        missing = [
            tool
            for tool, found in (("cmake", cmake), ("nvcc", nvcc))
            if not found
        ]
        if missing:
            joined = " and ".join(missing)
            raise SystemExit(
                f"{joined} was not found on PATH. flashtrie compiles a CUDA "
                "extension on the machine that will run it.\n"
                "Install the CUDA toolkit and CMake >= 3.14, then re-run "
                "pip install ."
            )

        ext_fullpath = Path(self.get_ext_fullpath(ext.name))
        extdir = ext_fullpath.parent.resolve()
        extdir.mkdir(parents=True, exist_ok=True)
        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DFLASHTRIE_PYTHON_OUTPUT_DIR={extdir}{os.sep}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DFLASHTRIE_BUILD_TOOLS=OFF",
            "-DFLASHTRIE_BUILD_PYTHON=ON",
        ]
        try:
            import pybind11
        except ImportError as exc:
            raise SystemExit(
                "pybind11 is required to build flashtrie. It is declared as a "
                "build dependency; re-run pip install . so the build "
                "environment can install it."
            ) from exc
        cmake_args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")

        if os.environ.get("CMAKE_ARGS"):
            cmake_args.extend(shlex.split(os.environ["CMAKE_ARGS"]))

        build_args = ["--config", cfg]
        jobs = os.environ.get("MAX_JOBS")
        if not jobs and getattr(self, "parallel", None):
            jobs = str(self.parallel)
        if jobs:
            build_args.append(f"-j{jobs}")

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [cmake, "-S", ext.sourcedir, "-B", str(build_temp), *cmake_args],
            check=True,
        )
        subprocess.run(
            [cmake, "--build", str(build_temp), *build_args],
            check=True,
        )
        if not ext_fullpath.exists():
            produced = sorted(extdir.glob("_core.*"))
            if not produced:
                produced = sorted(build_temp.rglob("_core*.so"))
                produced += sorted(build_temp.rglob("_core*.pyd"))
            if len(produced) != 1:
                raise RuntimeError(
                    f"expected the compiled extension at {ext_fullpath}, "
                    f"found {produced or 'nothing'}"
                )
            shutil.copy2(produced[0], ext_fullpath)


setup(
    ext_modules=[CMakeExtension("flashtrie._core", ".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
