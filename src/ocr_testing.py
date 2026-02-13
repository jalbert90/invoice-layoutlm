import json
from pathlib import Path
from paddleocr import PaddleOCR
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

def process_image(image_path):
    result = ocr.ocr(str(image_path))

    image = Image.open(image_path)
    width, height = image.size

    bboxes = []
    tokens = []

    if result and result[0]:
        for item in result[0]:
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
        'tokens': tokens,
        'bboxes': bboxes
    }

def main():
    
    input_dir = Path('data/images/batch1_1')
    output_dir = Path('data/ocr')
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = input_dir.glob('*')
    test_image_path = next(gen)

    data = process_image(test_image_path)

    output_file = output_dir / f'{test_image_path.stem}.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    main()
