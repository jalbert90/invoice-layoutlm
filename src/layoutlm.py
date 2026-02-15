import json
from pathlib import Path
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

ocr_dir = Path('data/ocr')
gen = ocr_dir.glob('*0062*')
test_ocr_path = next(gen)

with open(test_ocr_path, 'r') as f:
    ocr_data = json.load(f)

tokens = ocr_data['tokens']
bboxes = ocr_data['bboxes']
labels = ocr_data['labels']

label2id = {label: i for i, label in enumerate(set(labels))}
id2label = {i: label for label, i in label2id.items()}

# Tokenize words (OCR "tokens"), convert tokens to ids, duplicate bounding boxes,
# and preprocess images.
# processor = LayoutLMv3Processor.from_pretrained(
#     'microsoft/layoutlmv3-base',
#     apply_ocr=False
# )

# Load the base weigths.
# model = LayoutLMv3ForTokenClassification.from_pretrained(
#     'microsoft/layoutlmv3-base',
#     num_labels=len(label2id),
#     id2label=id2label,
#     label2id=label2id
# )
