import math

import argparse
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
import torch

from .layoutlm import InvoiceDataset
from .ocr import ocr_pipeline

def softmax(logits):
    s = 0
    for l in logits:
        sum += math.e ** l

    return [(math.e ** l) / sum for l in logits]

def infer(input_dir, ocr_save_dir, debug_dir):
    ocr_docs = ocr_pipeline(input_dir, ocr_save_dir, debug_dir=debug_dir)

    processor = LayoutLMv3Processor.from_pretrained(
        'microsoft/layoutlmv3-base',
        apply_ocr=False
    )    

    model = LayoutLMv3ForTokenClassification.from_pretrained('layoutlmv3-client/checkpoint-16')

    # Behaves like a list of dict[str, tensor]
    encoded_dataset = InvoiceDataset(ocr_docs, processor)
    sample = encoded_dataset[0]

    model.eval()

    with torch.no_grad():
        outputs = model(**sample)

    print(outputs.logits)
    print(outputs.logits.shape)

    probabilities = []
    for logits in outputs.logits[0]:
        probabilities.append(softmax(logits))

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
