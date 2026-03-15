import json
from paddleocr import PaddleOCR
from pathlib import Path

ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
input_dir = Path('data/B_training_images/curated')
output_dir = Path('data/ocr_raw')
output_dir.mkdir(parents=True, exist_ok=True)

gen = input_dir.glob('*Invoice_1.jpg*')
test_image_path = next(gen)

print('\n', f'Found {test_image_path.name}', '\n')

data = ocr.ocr(str(test_image_path))

output_file = output_dir / f'{test_image_path.stem}.json'
with open(output_file, 'w') as f:
    json.dump(data, f)

print(f'Saved {output_file}')
