#include <Python.h>

static PyObject* calculate_offsets(PyObject* self, PyObject* args) {
    int res_x, res_y, res_z;
    
    // Parse the input tuple (resolution)
    if (!PyArg_ParseTuple(args, "iii", &res_x, &res_y, &res_z)) {
        return NULL;
    }
    
    PyObject* offset_dict = PyDict_New();
    
    for (int x1 = 0; x1 < res_x; x1++) {
        for (int y1 = 0; y1 < res_y; y1++) {
            for (int z1 = 0; z1 < res_z; z1++) {
                for (int x2 = 0; x2 < res_x; x2++) {
                    for (int y2 = 0; y2 < res_y; y2++) {
                        for (int z2 = 0; z2 < res_z; z2++) {
                            int offset_x = abs(x1 - x2);
                            int offset_y = abs(y1 - y2);
                            int offset_z = abs(z1 - z2);
                            PyObject* offset_tuple = PyTuple_Pack(3, 
                                                PyLong_FromLong(offset_x), 
                                                PyLong_FromLong(offset_y), 
                                                PyLong_FromLong(offset_z));
                            
                            if (PyDict_Contains(offset_dict, offset_tuple) == 0) {
                                PyDict_SetItem(offset_dict, offset_tuple, PyLong_FromLong(PyDict_Size(offset_dict)));
                            }
                            Py_DECREF(offset_tuple);
                        }
                    }
                }
            }
        }
    }
    
    return offset_dict;
}


static PyObject* calculate_offsets_and_indices(PyObject* self, PyObject* args) {
    int res_x, res_y, res_z;
    
    // Parse the input tuple (resolution)
    if (!PyArg_ParseTuple(args, "iii", &res_x, &res_y, &res_z)) {
        return NULL;
    }
    
    PyObject* offset_dict = PyDict_New();
    PyObject* idxs_list = PyList_New(0);
    
    for (int x1 = 0; x1 < res_x; x1++) {
        for (int y1 = 0; y1 < res_y; y1++) {
            for (int z1 = 0; z1 < res_z; z1++) {
                for (int x2 = 0; x2 < res_x; x2++) {
                    for (int y2 = 0; y2 < res_y; y2++) {
                        for (int z2 = 0; z2 < res_z; z2++) {
                            int offset_x = abs(x1 - x2);
                            int offset_y = abs(y1 - y2);
                            int offset_z = abs(z1 - z2);
                            PyObject* offset_tuple = PyTuple_Pack(3, 
                                                PyLong_FromLong(offset_x), 
                                                PyLong_FromLong(offset_y), 
                                                PyLong_FromLong(offset_z));
                            
                            PyObject* idx;
                            if (PyDict_Contains(offset_dict, offset_tuple) == 0) {
                                idx = PyLong_FromLong(PyDict_Size(offset_dict));
                                PyDict_SetItem(offset_dict, offset_tuple, idx);
                            } else {
                                idx = PyDict_GetItem(offset_dict, offset_tuple);
                            }
                            PyList_Append(idxs_list, idx);
                            Py_DECREF(offset_tuple);
                        }
                    }
                }
            }
        }
    }
    
    return PyTuple_Pack(2, offset_dict, idxs_list);
}

static PyMethodDef OffsetMethods[] = {
    {"calculate_offsets", calculate_offsets, METH_VARARGS, "Calculate 3D offsets"},
    {"calculate_offsets_and_indices", calculate_offsets_and_indices, METH_VARARGS, "Calculate 3D offsets and indices"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef offsetmodule = {
    PyModuleDef_HEAD_INIT,
    "offsets",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    OffsetMethods
};

PyMODINIT_FUNC PyInit_offsets(void) {
    return PyModule_Create(&offsetmodule);
}
