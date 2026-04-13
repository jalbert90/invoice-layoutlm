import json
from pathlib import Path

import argparse
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
import torch

from .layoutlm import InvoiceDataset
from .ocr import ocr_pipeline

def infer(model_dir, input_dir, ocr_save_dir, debug_dir):
    ocr_docs = ocr_pipeline(input_dir, ocr_save_dir, debug_dir=debug_dir)

    image_paths = [Path(doc["image_path"]) for doc in ocr_docs]

    processor = LayoutLMv3Processor.from_pretrained(
        'microsoft/layoutlmv3-base',
        apply_ocr=False
    )    

    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
    model.eval()

    # Behaves like a list of dict[str, tensor]
    encoded_dataset = InvoiceDataset(ocr_docs, processor)

    client_names = {}

    print('\nRunning LayoutLM...')
    sample_num = 0
    for sample in encoded_dataset:
        print(f'\nProcessing {image_paths[sample_num].name}')
        with torch.no_grad():
            output = model(**sample)

        attention_mask = sample['attention_mask'][0]
        num_non_pad_tokens = attention_mask.sum().item()
        
        input_ids = sample['input_ids'][0]
        offset_mapping = sample['offset_mapping'][0][:num_non_pad_tokens].tolist()

        word_indices = []
        index_counter = -1
        for offset in offset_mapping:
            if offset == [0, 0]:
                word_indices.append(None)
                continue

            if offset[0] == 0:
                index_counter += 1
                word_indices.append(index_counter)
            else:
                word_indices.append(index_counter)

        logits = output.logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        client_name_id = model.config.label2id['client_name']
        pred_client_name_indices = torch.where(pred_ids == client_name_id)[0]
        pred_client_name_input_ids = input_ids[pred_client_name_indices]
        pred_client_name_tokens = processor.tokenizer.convert_ids_to_tokens(pred_client_name_input_ids)
        pred_client_name_token_word_indices = [word_indices[i] for i in pred_client_name_indices]

        valid_pred_client_name_indices = []
        valid_pred_client_name_input_ids = []
        valid_client_name_token_word_indices = []
        for i in pred_client_name_indices:
            i = i.item()
            if offset_mapping[i][0] == 0 and offset_mapping[i] != [0, 0]:
                valid_pred_client_name_indices.append(i)
                valid_pred_client_name_input_ids.append(input_ids[i])
                valid_client_name_token_word_indices.append(word_indices[i])

        valid_pred_client_name_tokens = processor.tokenizer.convert_ids_to_tokens(valid_pred_client_name_input_ids)

        print()
        print(pred_client_name_tokens)
        print(pred_client_name_token_word_indices)
        print(valid_pred_client_name_tokens)
        print(valid_client_name_token_word_indices)
        print()

        ocr_doc = ocr_docs[sample_num]
        pred_words = [ocr_doc['tokens'][i] for i in valid_client_name_token_word_indices]

        print(pred_words)
        print()

        client_names[image_paths[sample_num].name] = pred_words

        sample_num += 1

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
