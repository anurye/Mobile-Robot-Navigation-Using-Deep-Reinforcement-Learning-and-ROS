import os
import json
import yaml
import shutil


class DirectoryManager:
    def __init__(self, path):
        self.path = path

    def remove_if_present(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def create(self, exist_ok=True):
        os.makedirs(self.path, exist_ok=exist_ok)


def load_yaml(yaml_file_path):
    """Loads test configuration file"""
    with open(yaml_file_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(yaml_file_path, data):
    """Save data in a yaml file"""
    with open(yaml_file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def save_json(json_file_path, data):
    """Save data in a json file"""
    with open(json_file_path, "w") as file:
        json.dump(data, file, indent=4)
