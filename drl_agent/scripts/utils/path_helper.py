import os
import shutil


class DirectoryManager:

    def __init__(self, path):
        self.path = path
    
    def remove_if_present(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
    
    def create(self):
        os.makedirs(self.path)
