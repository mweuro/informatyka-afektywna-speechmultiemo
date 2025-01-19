import yaml
import importlib



def load_yaml(file_path: str) -> dict[dict[str, str]]:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def get_class_by_name(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)