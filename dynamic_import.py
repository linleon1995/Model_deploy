import os
from inspect import getmembers, isfunction, isclass
from inspect import isclass
import importlib
from pathlib import Path

from pprint import pprint


def find_class(module_path):
    """Find all the class in python file under module_path."""
    module_path = Path(module_path)
    if module_path.suffix == '':
        modules = list(module_path.rglob('*.py'))
    elif module_path.suffix == '.py':
        modules = [module_path]

    modules = [f for f in modules if f.name != '__init__.py']

    class_mapping = {}
    for module_path in modules:
        relative_path = os.path.relpath(module_path, os.getcwd())
        relative_path = Path(relative_path)

        dir_parts = relative_path.parts
        module_imp_name = '.'.join(dir_parts)[:-3]
        module = importlib.import_module(module_imp_name)
        classes = getmembers(module, isclass)
        for name, cls in classes:
            if cls.__module__.startswith(module_imp_name):
                class_mapping[f'{module_imp_name}.{name}'] = cls

    pprint(class_mapping)
    return class_mapping


if __name__ == '__main__':
    module_mapping = find_class('test_mod.model.py')
    module = module_mapping['test_mod.model.MockModel']
    model = module(v1=2, v2=3)
    print(model)
