from pathlib import Path

from paddleocr import PaddleOCR

def main():
    
    input_dir = Path('data/images/batch1_1')
    output_dir = Path('data/ocr')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generator items are yielded in disk order.
    for image_path in input_dir.glob('*'):
        print(f'Processing {image_path.name}')

if __name__ == '__main__':
    main()
