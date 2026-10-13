# Vendored Qhull 2019.1

Unmodified Qhull 2019.1 (`qhull_r 7.3.2 (2019.1.r 2019/06/21)`) source files, as bundled with SciPy 1.12.0
(`scipy/spatial/qhull_src/src`), redistributed under the Qhull licence in `COPYING.txt`.

INSIGHT's training graphs were triangulated with this Qhull compiled **without fused multiply-add**. Newer builds
(SciPy >= 1.13 wheels) contract multiply-adds, which changes how exactly cocircular points, such as the four corners of
a square of patches, are triangulated. `../_qhull_legacy.c` compiles these files with `-ffp-contract=off` so the SDK
reproduces the training graphs edge for edge on every platform.
