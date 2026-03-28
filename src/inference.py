import json
import math
from pathlib import Path

import argparse
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
import torch

from .layoutlm import InvoiceDataset
from .ocr import ocr_pipeline

def softmax(logits):
    s = 0
    for l in logits:
        s += math.e ** l

    return [(math.e ** l) / s for l in logits]

def infer(input_dir, ocr_save_dir, debug_dir):
    ocr_docs = ocr_pipeline(input_dir, ocr_save_dir, debug_dir=debug_dir)

    image_paths = [doc["image_path"] for doc in ocr_docs]

    processor = LayoutLMv3Processor.from_pretrained(
        'microsoft/layoutlmv3-base',
        apply_ocr=False
    )    

    model_dir = Path('layoutlmv3-client/checkpoint-16')
    model = LayoutLMv3ForTokenClassification.from_pretrained(str(model_dir))

    # Behaves like a list of dict[str, tensor]
    encoded_dataset = InvoiceDataset(ocr_docs, processor)
    sample_num = 0
    sample = encoded_dataset[sample_num]

    model.eval()

    with torch.no_grad():
        outputs = model(**sample)

    config_path = model_dir / f'config.json'
    with open(config_path) as f:
        config_data = json.load(f)

    print(f'\n\nImage Path:\n{image_paths[sample_num]}')

    print('\nIndex Labels:')
    print(config_data['id2label'], '\n')
    print('Logits:\n')
    print(outputs.logits.shape)
    print(outputs.logits)

    probabilities = []
    for logits in outputs.logits[0]:
        probabilities.append(softmax(logits))

    probabilities = torch.tensor(probabilities)
    print('\nProbabilities:\n')
    print(probabilities)

    index_0_label = config_data['id2label']['0']
    print(f'\nProb token is {index_0_label} | Token\n')

    count = 0
    for token in ocr_docs[sample_num]["tokens"]:
        print(round(probabilities[count][0].item(), 2), '\t', token)
        count += 1

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
