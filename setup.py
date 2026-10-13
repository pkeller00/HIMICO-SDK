"""Builds the vendored Qhull 2019.1 extension without fused multiply-add (see src/himico/_qhull/README.md)."""
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

QHULL = Path("src/himico/_qhull")


class BuildExt(build_ext):
    def build_extensions(self):
        flags = ["/fp:precise"] if self.compiler.compiler_type == "msvc" else ["-ffp-contract=off", "-w"]
        for ext in self.extensions:
            ext.extra_compile_args = flags
        super().build_extensions()


setup(
    ext_modules=[Extension(
        "himico._qhull_legacy",
        sources=["src/himico/_qhull_legacy.c"] + sorted(str(p) for p in QHULL.glob("*.c")),
        include_dirs=[str(QHULL)],
    )],
    cmdclass={"build_ext": BuildExt},
)
