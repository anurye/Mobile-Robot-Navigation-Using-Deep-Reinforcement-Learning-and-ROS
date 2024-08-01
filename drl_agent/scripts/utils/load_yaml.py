import sys
import yaml


def load_yaml_file(yaml_file_path):
    """Loads test configuration file"""
    try:
        with open(yaml_file_path, 'r') as file:
            yaml_file = yaml.safe_load(file)
    except Exception as e:
        print(f"Unable to load: {yaml_file_path}: {e}")
        sys.exit(-1)

    return yaml_file


if __name__=="__main__":
    pass
