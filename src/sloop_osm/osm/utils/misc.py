import importlib

def import_class(class_string):
    """Returns the class pointed to by 'class_string',
    which looks something like 'module1.sub1.Class1'"""
    parts = class_string.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def tobeoverriden(f):
    return f
