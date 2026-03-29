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

def infer(model_dir, input_dir, ocr_save_dir, debug_dir):
    ocr_docs = ocr_pipeline(input_dir, ocr_save_dir, debug_dir=debug_dir)

    image_paths = [doc["image_path"] for doc in ocr_docs]

    processor = LayoutLMv3Processor.from_pretrained(
        'microsoft/layoutlmv3-base',
        apply_ocr=False
    )    

    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)

    # Behaves like a list of dict[str, tensor]
    encoded_dataset = InvoiceDataset(ocr_docs, processor)
    
    print('\n\nSample Number | Sample Path\n')
    for i in range(len(image_paths)):
        print(f'{i}\t{image_paths[i]}')

    sample_num = int(input('Enter sample number: '))
    sample = encoded_dataset[sample_num]

    model.eval()

    with torch.no_grad():
        outputs = model(**sample)

    model_dir = Path(model_dir)
    config_path = model_dir / f'config.json'
    with open(config_path) as f:
        config_data = json.load(f)

    client_name_index = config_data['label2id']['client_name']

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
    print(probabilities.shape)
    print(probabilities)

    probs = torch.softmax(outputs.logits, dim=-1)

    print('\n\nBetter probs:\n')
    print(probs.shape)
    print(probs)

    pred_ids = outputs.logits.argmax(dim=-1)

    print(f'\nProb token is client_name | Token\n')

    count = 0
    for token in ocr_docs[sample_num]["tokens"]:
        print(round(probabilities[count][client_name_index].item(), 2), '\t', token)
        count += 1

    client_names = {}

    return client_names

if __name__ == '__main__':
    MODEL_DIR = 'models/batch1_1/checkpoint-95'

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/3_inference_pipeline/1_images/default')
    parser.add_argument('--ocr_save_dir', type=str, default='data/3_inference_pipeline/2_ocr/default')
    parser.add_argument('--debug_dir', type=str, default=None)
    args = parser.parse_args()

    client_names = infer(MODEL_DIR, args.input_dir, args.ocr_save_dir, args.debug_dir)

    for invoice_name, client_name in client_names.items():
        print(f'{invoice_name}: \t {client_name}')
