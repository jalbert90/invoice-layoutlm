import argparse
import cv2
import json
import numpy as np
from paddleocr import PaddleOCR
from pathlib import Path
from PIL import Image

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def normalize_bbox(bbox, width, height):
    """
    Convert absolute bbox to LayoutLM normalized bbox (0-1000 scale).

    :param bbox: [x_min, y_min, x_max, y_max]
    :param width: The width of the containing image.
    :param height: The height of the containing image.
    :return: [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
    """
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height)
    ]

def polygon_to_bbox(polygon):
    """
    Convert a polygon defined by 4 points to a rectangular box defined by 2 points.
    
    :param polygon: [[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]]
    :return: [x_min, y_min, x_max, y_max]
    """
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]

def process_raw_ocr(raw_ocr, image_path):
    image = Image.open(image_path)
    width, height = image.size

    bboxes = []
    tokens = []

    if raw_ocr and raw_ocr[0]:
        for item in raw_ocr[0]:
            polygon = item[0]
            text, confidence = item[1]

            # Skip empty boxes
            if text.strip() == '':
                continue

            bbox = polygon_to_bbox(polygon)
            bbox_norm = normalize_bbox(bbox, width, height)

            bboxes.append(bbox_norm)
            tokens.append(text)

    return {
        'image_path': str(image_path),
        'tokens': tokens,
        'bboxes': bboxes
    }

def visualize_ocr(image_path, raw_ocr, output_dir):
    image = cv2.imread(str(image_path))

    for item in raw_ocr[0]:
        bbox = np.array(item[0]).astype(np.int32)
        cv2.polylines(image, [bbox], isClosed=True, color=(0,255,0))

    output_path = output_dir / f'{image_path.stem}-overlay.jpg'
    cv2.imwrite(str(output_path), image)

def ocr_pipeline(input_dir, output_dir, debug_dir=None) -> list[dict[str, str]]:
    """Returns OCR docs."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        ocr_raw_dir = debug_dir / f'ocr_raw/{input_dir.name}'
        ocr_raw_dir.mkdir(parents=True, exist_ok=True)
        ocr_raw_visualize_dir = debug_dir / f'ocr_raw_visualize/{input_dir.name}'
        ocr_raw_visualize_dir.mkdir(parents=True, exist_ok=True)

    ocr_docs = []

    # Generator items are yielded in disk order.
    for image_path in input_dir.glob('*'):
        print(f'\n\nProcessing {image_path.name}')

        raw_ocr = ocr.ocr(str(image_path))
        processed_ocr = process_raw_ocr(raw_ocr, image_path)
        ocr_docs.append(processed_ocr)
        print()

        if debug_dir:
            ocr_raw_output_file = ocr_raw_dir / f'{image_path.stem}.json'
            with open(ocr_raw_output_file, 'w') as f:
                json.dump(raw_ocr, f, indent=2)
            
            visualize_ocr(image_path, raw_ocr, ocr_raw_visualize_dir)
        
        output_file = output_dir / f'{image_path.stem}.json'
        with open(output_file, 'w') as f:
            json.dump(processed_ocr, f, indent=2)

    return ocr_docs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/2_training_pipeline/1_images/default')
    parser.add_argument('--output_dir', type=str, default='data/2_training_pipeline/2_ocr/default')
    parser.add_argument('--debug_dir', type=str, default=None)
    args = parser.parse_args()

    ocr_pipeline(args.input_dir, args.output_dir, args.debug_dir)
