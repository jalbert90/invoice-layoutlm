from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# Tokenize words (OCR "tokens"), convert tokens to ids, duplicate bounding boxes,
# and preprocess images.
processor = LayoutLMv3Processor.from_pretrained(
    'microsoft/layoutlmv3-base',
    apply_ocr=False
)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    'microsoft/layoutlmv3-base',
    # num_labels=?
)
