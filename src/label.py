import json
from pathlib import Path

OCR_DIR = 'data/2_training_pipeline/2_ocr/batch1_1'

ocr_dir = Path(OCR_DIR)

for ocr_path in ocr_dir.glob('*'):
    print(f'Labeling {ocr_path.name}')

    with open(ocr_path, 'r') as f:
        data = json.load(f)

    data['labels'] = ['0' for token in data['tokens']]
    data['labels'][6] = 'client_name'

    data['image_path'] = str(ocr_dir.parent.parent / f'1_images/batch1_1/{ocr_path.stem}.jpg')

    with open(ocr_path, 'w') as f:
        json.dump(data, f, indent=2)
