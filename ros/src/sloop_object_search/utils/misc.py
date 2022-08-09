import importlib
import hashlib

def import_class(class_string):
    """Returns the class pointed to by 'class_string',
    which looks something like 'module1.sub1.Class1'"""
    parts = class_string.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def import_func(func_string):
    """Returns the function pointed to by 'func_string',
    which looks something like 'module1.sub1.func1'"""
    # works the same way as import class
    return import_class(func_string)


def det_dict_hash(dct, keep=9):
    """deterministic hash of a dictionary."""
    content = str(list(sorted(dct.items()))).encode()
    hashcode = int(str(int(hashlib.sha1(content).hexdigest(), 16))[:keep])
    return hashcode
