# Next steps:

1. Take highest probability prediction. (SKIP)
  - Bad idea. The client name may span multiple ocr boxes, in which case taken the highest prob prediction will only capture a partial client name. Instead, focus on better training.
2. Implement metrics.
3. Reorganize data -> dataset_name -> train/test -> etc
4. Retrain, possibly expanding context width.
5. Clean up inference output and make it save.

# Ideas for later:

- Change `ocr_pipeline()` to use save and debug flags so that saving isn't strictly necessary.
- Change pipeline format so that an input directory is taken and an output directory is taken, then all of the outputs (ocr, ocr_raw, ocr_raw_visualize, client_names) are all stored in the output directory. Example:

```plaintext
data/
  |--inference_pipeline/
    |--input_dirs/
      |--dataset_name/
    |--output_dirs/
      |--dataset_name_outputs
```

...maybe... refine this before making any more changes directory changes...

- Organize by dataset so that image path can always be found when hardcoded into the ocr data.

- Change image path in ocr data to image name.

- inference.py should take an input directory and output directory only. Output directory should be optional. Can later add output directory verbosity levels.

- Fix the crazy long names in inference.py.

# Environment Notes:

- Needed to install apt package swig on LT WSL2 in order to build the PyMuPDF wheel from source, which was needed to install paddleocr 2.7.0.3.
- pip install -r requirements.txt should work...

# Math Notes:

# Deprecated. Keeping for math reference only.
# def softmax(logits):
#     s = 0
#     for l in logits:
#         s += math.e ** l

#     return [(math.e ** l) / s for l in logits]

# Pipeline notes

- OCR creates words, which are then labeled.
- LayoutLMv3 then splits the words into tokens.
  - The labels are not spread across the tokens by default.
  - The first token of each word is given an integer label that matches the OCR label (as specified in label2id).
  - The remaining tokens of each word are given the label -100, which means ignore during loss calculation.
- Then, LayoutLM token predictions should be mapped back to an OCR word, trusting only the first token of each word.
