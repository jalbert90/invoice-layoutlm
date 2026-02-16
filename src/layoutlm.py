import json
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

class InvoiceDataset(Dataset):
    def __init__(self, docs, processor, label2id):
        self.docs = docs
        self.processor = processor
        self.label2id = label2id
    
    def __len__(self):
        return len(self.docs)
    
    def __getitem__(self, idx):
        item = self.docs[idx]

        image_path = item['image_path']
        tokens = item['tokens']
        bboxes = item['bboxes']
        labels = [self.label2id[label] for label in item['labels']]

        encoding = self.processor(
            images=image_path,
            text=tokens,
            boxes=bboxes,
            word_labels=labels,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

ocr_dir = Path('data/ocr')
docs = []

for ocr_path in ocr_dir.glob('*'):
    with open(ocr_path, 'r') as f:
        doc = json.load(f)

    docs.append(doc)

label2id = {label: i for i, label in enumerate(set(labels))}
id2label = {i: label for label, i in label2id.items()}

# test_image = Image.open(test_image_path).convert('RGB')

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

# encoding = processor(
#     images=test_image,
#     text=tokens,
#     boxes=bboxes,
#     word_labels=[label2id[label] for label in labels],
#     padding='max_length',
#     truncation=True,
#     max_length=512,
#     return_tensors='pt'
# )

# outputs = model(**encoding)

# print(outputs.loss)
# print(outputs.logits.shape)
