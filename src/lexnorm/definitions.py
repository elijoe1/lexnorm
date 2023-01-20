import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "../../data")
LEX_PATH = os.path.join(DATA_PATH, "external/scowl-2020.12.07/final")

if __name__ == "__main__":
    print(DATA_PATH)
