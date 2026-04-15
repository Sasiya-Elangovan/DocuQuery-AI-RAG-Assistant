import os

class Config:
    PERSIST_DIR = "docs/chroma"
    UPLOAD_DIR = "data"

    def __init__(self):
        os.makedirs(self.PERSIST_DIR, exist_ok=True)
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

config = Config()