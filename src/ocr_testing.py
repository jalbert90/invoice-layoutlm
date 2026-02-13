from pathlib import Path
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def process_image(image_path):
    result = ocr.ocr(str(image_path))
    print('\n\n', image_path.name, '\n')

    if result and result[0]:
        for item in result[0]:
            polygon = item[0]
            text, confidence = item[1]

            # Skip empty boxes
            if text.strip() == '':
                continue

            print(f'{polygon} \t {confidence} \t {text}')

def main():
    
    input_dir = Path('data/images/batch1_1')
    output_dir = Path('data/ocr')
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = input_dir.glob('*')
    test_image_path = next(gen)

    process_image(test_image_path)

if __name__ == '__main__':
    main()
