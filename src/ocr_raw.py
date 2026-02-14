import json
from paddleocr import PaddleOCR
from pathlib import Path

ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
input_dir = Path('data/images/batch1_1')
output_dir = Path('data/ocr_raw')
output_dir.mkdir(parents=True, exist_ok=True)

gen = input_dir.glob('*0062*')
test_image_path = next(gen)

data = ocr.ocr(str(test_image_path))

output_file = output_dir / f'{test_image_path.stem}.json'
with open(output_file, 'w') as f:
    json.dump(data, f)

print(data)
