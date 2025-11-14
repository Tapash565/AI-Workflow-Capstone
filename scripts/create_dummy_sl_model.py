import os
from pathlib import Path
import joblib
from scripts.dummy_model import DummyModel


def main():
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    file_path = models_dir / 'sl-all-0_1.joblib'
    model = DummyModel()
    joblib.dump(model, file_path)
    print('Wrote', file_path)


if __name__ == '__main__':
    main()
