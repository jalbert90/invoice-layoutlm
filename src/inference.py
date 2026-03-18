import argparse
from transformers import LayoutLMv3ForTokenClassification

from .ocr import ocr_pipeline

def infer(input_dir, ocr_save_dir, debug_dir):
    ocr_docs = ocr_pipeline(input_dir, ocr_save_dir, debug_dir=debug_dir)

    LayoutLMv3ForTokenClassification.from_pretrained('layoutlmv3-client/checkpoint-16')

    client_names = {}

    return client_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/3_inference_pipeline/1_images/default')
    parser.add_argument('--ocr_save_dir', type=str, default='data/3_inference_pipeline/2_ocr/default')
    parser.add_argument('--debug_dir', type=str, default=None)
    args = parser.parse_args()

    client_names = infer(args.input_dir, args.ocr_save_dir, args.debug_dir)

    for invoice_name, client_name in client_names.items():
        print(f'{invoice_name}: \t {client_name}')
