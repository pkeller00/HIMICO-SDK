/*
 * 2-D Delaunay triangulation with the vendored Qhull 2019.1 (see _qhull/README.md), called exactly as
 * scipy.spatial.Delaunay does ("qhull d Qbb Qc Qz Q12 Qt"). Compiled without fused multiply-add so that ties between
 * cocircular points are broken as in the graphs INSIGHT was trained on.
 *
 *   delaunay(points: bytes-like of float64, n x 2, C order) -> bytes of int32, n_simplices x 3
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "_qhull/qhull_ra.h"

static PyObject *delaunay(PyObject *self, PyObject *args) {
    Py_buffer buf;
    if (!PyArg_ParseTuple(args, "y*", &buf)) return NULL;
    if (buf.len % (2 * sizeof(coordT)) != 0) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError, "points must be float64 with shape (n, 2)");
        return NULL;
    }
    int n = (int)(buf.len / (2 * sizeof(coordT)));
    if (n < 3) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError, "need at least 3 points");
        return NULL;
    }
    /* Qhull may modify its input (Qbb scales the last coordinate), so give it a copy */
    coordT *pts = (coordT *)malloc(buf.len);
    if (!pts) { PyBuffer_Release(&buf); return PyErr_NoMemory(); }
    memcpy(pts, buf.buf, buf.len);
    PyBuffer_Release(&buf);

    qhT qh_qh;
    qhT *qh = &qh_qh;
    char cmd[] = "qhull d Qbb Qc Qz Q12 Qt";
    PyObject *result = NULL;
    int code;
    Py_BEGIN_ALLOW_THREADS
    qh_zero(qh, stderr);
    code = qh_new_qhull(qh, 2, n, pts, False, cmd, NULL, stderr);
    Py_END_ALLOW_THREADS
    if (code) {
        PyErr_Format(PyExc_RuntimeError, "qhull failed with code %d (points may be degenerate)", code);
    } else {
        facetT *facet;
        vertexT *vertex, **vertexp;
        Py_ssize_t count = 0;
        FORALLfacets { if (!facet->upperdelaunay) count++; }
        result = PyBytes_FromStringAndSize(NULL, count * 3 * sizeof(int));
        if (result) {
            int *out = (int *)PyBytes_AS_STRING(result);
            Py_ssize_t k = 0;
            FORALLfacets {
                if (facet->upperdelaunay) continue;
                FOREACHvertex_(facet->vertices) { out[k++] = qh_pointid(qh, vertex->point); }
            }
        }
    }
    int curlong, totlong;
    qh_freeqhull(qh, !qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    free(pts);
    return result;
}

static PyMethodDef methods[] = {
    {"delaunay", delaunay, METH_VARARGS, "Delaunay simplices (int32 n x 3) of float64 n x 2 points, Qhull 2019.1."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_qhull_legacy", NULL, -1, methods};

PyMODINIT_FUNC PyInit__qhull_legacy(void) { return PyModule_Create(&module); }
