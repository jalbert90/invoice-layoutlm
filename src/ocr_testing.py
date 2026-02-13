from pathlib import Path
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def process_image(image_path):
    result = ocr.ocr(str(image_path))
    print(result)

def main():
    
    input_dir = Path('data/images/batch1_1')
    output_dir = Path('data/ocr')
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = input_dir.glob('*')
    test_image = next(gen)

    process_image(test_image)

if __name__ == '__main__':
    main()
