Ideas for later:

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
