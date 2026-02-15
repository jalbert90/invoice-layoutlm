import cv2
import json
import numpy as np
from pathlib import Path

def visualize_ocr(image_path, ocr_raw_path, output_dir='data/visualize_ocr'):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))

    with open(str(ocr_raw_path), 'r') as f:
        data = json.load(f)

    for line in data[0]:
        pts = np.array(line[0]).astype(np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0))

    output_path = output_dir / f'{image_path.stem}-overlay.jpg'
    cv2.imwrite(str(output_path), image)

    print(f'Visualization saved as {output_path}')

def main():
    visualize_ocr('data/images/batch1_1/batch1-0062.jpg', 'data/ocr_raw/batch1-0062.json')

if __name__ == '__main__':
    main()
