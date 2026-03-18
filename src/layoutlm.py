from itertools import islice
import json
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, TrainingArguments, Trainer

class InvoiceDataset(Dataset):
    def __init__(self, docs, processor, training=False, label2id=None):
        """Must pass label2id if training=True."""
        self.docs = docs
        self.processor = processor
        self.training = training
        self.label2id = label2id
    
    def __len__(self):
        return len(self.docs)
    
    def __getitem__(self, idx):
        doc = self.docs[idx]

        image_path = doc['image_path']
        tokens = doc['tokens']
        bboxes = doc['bboxes']

        image = Image.open(image_path).convert('RGB')

        if self.training:
            labels = [self.label2id[label] for label in doc['labels']]

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
            # Strip batch dimension for Trainer
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        else:
            encoding = self.processor(
                images=image,
                text=tokens,
                boxes=bboxes,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

        return encoding
    
def label2id(ocr_dict):
    # I need to make this deterministic so that I know what the output of the logits tensor means!!!!!!!!!!!!!!!
    return {label: i for i, label in enumerate(set(ocr_dict['labels']))}

def main():
    ocr_dir = Path('data/C_ocr/curated')
    docs = []

    for ocr_path in islice(ocr_dir.glob('*'), 100):
        with open(ocr_path, 'r') as f:
            doc = json.load(f)

        docs.append(doc)

    label2id = label2id(docs[0])
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

    train_docs, val_docs = train_test_split(docs, test_size=0.25, random_state=5)

    train_dataset = InvoiceDataset(train_docs, processor, label2id)
    val_dataset = InvoiceDataset(val_docs, processor, label2id)

    training_args = TrainingArguments(
        output_dir='layoutlmv3-client',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=7,
        eval_strategy='epoch',
        save_strategy='best',
        metric_for_best_model='eval_loss',
        save_total_limit=1,
        fp16=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=1,
        remove_unused_columns=False
    )

    # Trainer takes tensors without batch dimension because it uses a data loader internally.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

if __name__ == '__main__':
    main()
