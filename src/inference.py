import json
from pathlib import Path

import argparse
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
import torch

from .layoutlm import InvoiceDataset
from .ocr import ocr_pipeline

def infer(model_dir, input_dir, ocr_save_dir, debug_dir):
    ocr_docs = ocr_pipeline(input_dir, ocr_save_dir, debug_dir=debug_dir)

    image_paths = [doc["image_path"] for doc in ocr_docs]

    processor = LayoutLMv3Processor.from_pretrained(
        'microsoft/layoutlmv3-base',
        apply_ocr=False
    )    

    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
    model.eval()

    # Behaves like a list of dict[str, tensor]
    encoded_dataset = InvoiceDataset(ocr_docs, processor)

    print()
    sample_num = 0
    for sample in encoded_dataset:
        print(f'Running LayoutLM on {image_paths[sample_num]}')
        with torch.no_grad():
            output = model(**sample)
        
        input_ids = sample['input_ids'][0]

        logits = output.logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        client_name_id = model.config.label2id['client_name']
        pred_client_name_indices = torch.where(pred_ids == client_name_id)[0]
        pred_client_name_input_ids = input_ids[pred_client_name_indices]
        pred_client_name_tokens = processor.tokenizer.convert_ids_to_tokens(pred_client_name_input_ids)
        print(pred_client_name_tokens)

        sample_num += 1

    client_names = {}

    return client_names

if __name__ == '__main__':
    MODEL_DIR = 'models/curated/checkpoint-16'

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/3_inference_pipeline/1_images/default')
    parser.add_argument('--ocr_save_dir', type=str, default='data/3_inference_pipeline/2_ocr/default')
    parser.add_argument('--debug_dir', type=str, default=None)
    args = parser.parse_args()

    client_names = infer(MODEL_DIR, args.input_dir, args.ocr_save_dir, args.debug_dir)

    for invoice_name, client_name in client_names.items():
        print(f'{invoice_name}: \t {client_name}')
