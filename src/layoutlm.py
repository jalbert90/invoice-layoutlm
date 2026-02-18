import json
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, TrainingArguments, Trainer

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
            max_length=512,
            return_tensors='pt'
        )

        # Strip batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        return encoding

ocr_dir = Path('data/ocr')
docs = []

for ocr_path in ocr_dir.glob('*'):
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
model = LayoutLMv3ForTokenClassification.from_pretrained(
    'microsoft/layoutlmv3-base',
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

train_docs, val_docs = train_test_split(docs, test_size=0.75, random_state=5)

train_dataset = InvoiceDataset(train_docs, processor, label2id)
val_dataset = InvoiceDataset(val_docs, processor, label2id)

training_args = TrainingArguments(
    output_dir='layoutlmv3-client',
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
