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

    # Behaves like a list of dict[str, tensor]
    encoded_dataset = InvoiceDataset(ocr_docs, processor)
    
    print('\n\nSample Number | Sample Path\n')
    for i in range(len(image_paths)):
        print(f'{i}\t{image_paths[i]}')

    sample_num = int(input('Enter sample number: '))
    sample = encoded_dataset[sample_num]
    input_ids = sample['input_ids']
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[0])

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

    probabilities = torch.softmax(outputs.logits, dim=-1)

    num_of_tokens = probabilities.shape[1]
    for i in range(num_of_tokens):
        prob_vector = probabilities[0][i].tolist()
        prob_vector = [round(p, 2) for p in prob_vector]
        token = tokens[i]        
        print(f'{prob_vector}\t{token}')

    # print('Logits:\n')
    # print(outputs.logits.shape)
    # print(outputs.logits)

    # print('\nProbabilities:\n')
    # print(probabilities.shape)
    # print(probabilities)

    client_name_token_ids = []
    token_index = 0
    for v in probabilities[0]:
        if v[client_name_index] >= 0.75:
            client_name_token_ids.append(input_ids[0][token_index].item())
        token_index += 1

    print('\n', client_name_token_ids)
    client_name_tokens = processor.tokenizer.decode(client_name_token_ids, skip_special_tokens=True)
    print(client_name_tokens)

    logits = outputs.logits
    pred_ids = logits.argmax(dim=-1)[0]
    print('\n', pred_ids)

    client_label_id = model.config.label2id['client_name']
    predicted_indicies = torch.where(pred_ids == client_label_id)[0]
    print(predicted_indicies)

    predicted_ids = input_ids[0][predicted_indicies]
    predicted_tokens = processor.tokenizer.convert_ids_to_tokens(predicted_ids)
    print(predicted_tokens)

    predicted_tokens_decoded = processor.tokenizer.decode(predicted_ids, skip_special_tokens=True)
    print(predicted_tokens_decoded)

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
