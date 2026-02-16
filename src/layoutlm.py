import json
from pathlib import Path
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

ocr_dir = Path('data/ocr')
images_dir = Path('data/images/batch1_1')
ocr_gen = ocr_dir.glob('*0062*')
images_gen = images_dir.glob('*0062*')
test_ocr_path = next(ocr_gen)
test_image_path = next(images_gen)

with open(test_ocr_path, 'r') as f:
    ocr_data = json.load(f)

tokens = ocr_data['tokens']
bboxes = ocr_data['bboxes']
labels = ocr_data['labels']

label2id = {label: i for i, label in enumerate(set(labels))}
id2label = {i: label for label, i in label2id.items()}

test_image = Image.open(test_image_path).convert('RGB')

# Tokenize words (OCR "tokens"), convert tokens to ids, duplicate bounding boxes,
# and preprocess images.
processor = LayoutLMv3Processor.from_pretrained(
    'microsoft/layoutlmv3-base',
    apply_ocr=False
)

# Load the base weigths.
model = LayoutLMv3ForTokenClassification.from_pretrained(
    'microsoft/layoutlmv3-base',
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

encoding = processor(
    images=test_image,
    text=tokens,
    boxes=bboxes,
    word_labels=[label2id[label] for label in labels],
    padding='max_length',
    truncation=True,
    max_length=512,
    return_tensors='pt'
)

print(encoding)
