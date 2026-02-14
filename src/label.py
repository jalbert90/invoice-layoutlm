import json
from pathlib import Path

ocr_dir = Path('data/ocr')

for json_path in ocr_dir.glob('*'):
    print(f'Labeling {json_path}')

    with open(json_path, 'r') as f:
        data = json.load(f)

    data['labels'] = ['0' for token in data['tokens']]
    data['labels'][5] = 'seller_name'

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
