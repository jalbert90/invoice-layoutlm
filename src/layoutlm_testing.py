from itertools import islice
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

OCR_DIR = 'data/2_training_pipeline/2_ocr/default'

class InvoiceDataset(Dataset):
    def __init__(self, docs, processor, label2id):
        self.docs = docs
        self.processor = processor
        self.label2id = label2id
    
    def __len__(self):
        return len(self.docs)
    
    def __getitem__(self, idx):
        doc = self.docs[idx]

        image_path = doc['image_path']
        tokens = doc['tokens']
        bboxes = doc['bboxes']
        labels = [self.label2id[label] for label in doc['labels']]

        image = Image.open(image_path).convert('RGB')

        encoding = self.processor(
            images=image,
            text=tokens,
            boxes=bboxes,
            word_labels=labels,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        # Strip batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        return encoding

ocr_dir = Path(OCR_DIR)
docs = []

gen = ocr_dir.glob('*')

# for ocr_path in islice(gen, 5):
#     with open(ocr_path, 'r') as f:
#         doc = json.load(f)
#     docs.append(doc)

test_ocr_path = next(gen)

with open(test_ocr_path, 'r') as f:
    test_doc = json.load(f)

test_image = Image.open(test_doc['image_path']).convert('RGB')

for ocr_path in islice(ocr_dir.glob('*'), 5):
    with open(ocr_path, 'r') as f:
        doc = json.load(f)

    docs.append(doc)

label2id = {label: i for i, label in enumerate(set(docs[0]['labels']))}
id2label = {i: label for label, i in label2id.items()}

# Tokenize words (OCR "tokens"), convert tokens to ids, duplicate bounding boxes,
# and preprocess images.
processor = LayoutLMv3Processor.from_pretrained(
    'microsoft/layoutlmv3-base',
    apply_ocr=False
)

# Load the base weigths.
# model = LayoutLMv3ForTokenClassification.from_pretrained(
#     'microsoft/layoutlmv3-base',
#     num_labels=len(label2id),
#     id2label=id2label,
#     label2id=label2id
# )

encoding = processor(
            images=test_image,
            text=test_doc['tokens'],
            boxes=test_doc['bboxes'],
            word_labels=[label2id[label] for label in test_doc['labels']],
            padding='max_length',
            truncation=True,
            # max_length=128,
            return_tensors='pt'
        )

# ids = encoding['input_ids'][0].tolist()
# attention = encoding['attention_mask'][0]
# print(attention.sum().item())

# print(ids)

# tokens = processor.tokenizer.convert_ids_to_tokens(ids)

# print(tokens)

# decoded = processor.tokenizer.decode(ids, skip_special_tokens=True)

# print(decoded)

# data = InvoiceDataset(docs, processor, label2id)

tokens = processor.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
labels = encoding['labels'][0]

for t, l in zip(tokens, labels):
    print(f'{l.item()}\t{t}')

# print(encoding)
# print(data[0])

# for k, v in encoding.items():
#     print(k, v.shape)

# for k, v in data[0].items():
#     print(k, v.shape)
