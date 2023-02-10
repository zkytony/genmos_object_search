#ifndef C_API_UTILS_H
#define C_API_UTILS_H

#include <Python.h>
#include <numpy/arrayobject.h>

// must do this (numpy c api mandate)
// also must do this in a separate function
PyMODINIT_FUNC
init_numpy(){
    import_array(); // PyError if not successful
    return 0;
}

///// The following functions are actually from SWIG https://swig.org/
/*
 * Given a PyObject pointer, cast it to a PyArrayObject pointer if
 * legal.  If not, set the python error string appropriately and
 * return NULL.
 *
 * SOURCE: https://github.com/PMBio/peer/blob/master/python/swig_typemaps.i#L101
 */
PyArrayObject* obj_to_array_no_conversion(PyObject* input, int typecode);

// Returns true if PyObject is a numpy array
// source: https://github.com/PMBio/peer/blob/master/python/swig_typemaps.i#L31
bool is_array(PyObject* a) { return (a) && PyArray_Check(a); }
int array_type(PyObject* a) { return (int) PyArray_TYPE(a); }

// Given a PyObject, return a string describing its type.
// source: https://github.com/PMBio/peer/blob/master/python/swig_typemaps.i#L39
// note: needed to fix (e.g. PyString_Check to PyUnicode_Check) for python 3
//       PyInstance_Check and PyFile_Check are removed
const char* typecode_string(PyObject* py_obj) {
  if (py_obj == NULL          ) return "C NULL value";
  if (PyCallable_Check(py_obj)) return "callable"    ;
  if (PyUnicode_Check(  py_obj)) return "string"      ;
  if (PyLong_Check(     py_obj)) return "int"         ;
  if (PyFloat_Check(   py_obj)) return "float"       ;
  if (PyDict_Check(    py_obj)) return "dict"        ;
  if (PyList_Check(    py_obj)) return "list"        ;
  if (PyTuple_Check(   py_obj)) return "tuple"       ;
  if (PyModule_Check(  py_obj)) return "module"      ;
  return "unknown type";
}

/* Given a numpy typecode, return a string describing the type, assuming
the following numpy type codes (this is already defined in numpy/arrayobject.h)
source: https://github.com/PMBio/peer/blob/master/python/swig_typemaps.i#L31
enum NPY_TYPES {    NPY_BOOL=0,
                    NPY_BYTE, NPY_UBYTE,
                    NPY_SHORT, NPY_USHORT,
                    NPY_INT, NPY_UINT,
                    NPY_LONG, NPY_ULONG,
                    NPY_LONGLONG, NPY_ULONGLONG,
                    NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
                    NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
                    NPY_OBJECT=17,
                    NPY_STRING, NPY_UNICODE,
                    NPY_VOID,
                    NPY_NTYPES,
                    NPY_NOTYPE,
                    NPY_CHAR,
                    NPY_USERDEF=256
 */

// source: https://github.com/PMBio/peer/blob/master/python/swig_typemaps.i#L31
const char* typecode_string(int typecode) {
    const char* type_names[24] = {"bool","byte","unsigned byte","short",
        "unsigned short","int","unsigned int","long",
        "unsigned long","long long", "unsigned long long",
        "float","double","long double",
        "complex float","complex double","complex long double",
        "object","string","unicode","void","ntype","notype","char"};
    const char* user_def="user defined";

    if (typecode>24)
        return user_def;
    else
        return type_names[typecode];
}

// converts PyObject to PyArrayObject (a numpy type)
// somehow this function throws segfault when placed in the .cpp file;
// This has something to do with numpy c api's import_array() business,
// but I still get segfault even if I call import_array in the .cpp file.
PyArrayObject* obj_to_array_no_conversion(PyObject* input, int typecode) {
    PyArrayObject* ary = NULL;
    bool match = PyArray_EquivTypenums(array_type(input),
                                       typecode);
    if (is_array(input) && (typecode == PyArray_NOTYPE || match)) {
        ary = (PyArrayObject*) input;
    } else if (is_array(input)) {
        const char* desired_type = typecode_string(typecode);
        const char* actual_type = typecode_string(array_type(input));
        PyErr_Format(PyExc_TypeError,
                     "Array of type '%s' required.  Array of type '%s' given",
                     desired_type, actual_type);
        ary = NULL;
    } else {
        const char* desired_type = typecode_string(typecode);
        const char* actual_type = typecode_string(input);
        PyErr_Format(PyExc_TypeError,
                     "Array of type '%s' required.  Array of type '%s' given",
                     desired_type, actual_type);
        ary = NULL;
    }
    return ary;
}



#endif
