import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data/external")

if __name__ == "__main__":
    print(DATA_PATH)
