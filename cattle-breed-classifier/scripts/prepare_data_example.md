# Expected data layout (folder-per-breed)

Point `config.yaml -> data.data_dir` (or the notebook's `DATA_DIR`) at a folder
structured like this:

    data/
      Gir/
        gir_001.jpg
        gir_002.jpg
      Sahiwal/
        sahiwal_001.jpg
      Tharparkar/
        ...
      Red_Sindhi/
        ...

- The sub-folder name becomes the class label.
- No CSV needed. Any number of breeds is supported.
- Roughly balanced classes train best; if very imbalanced, gather more images
  for the small classes or the confusion matrix will show it.

If you already have fixed train/val/test folders, set `data.has_split: true`
and use this layout instead:

    data/
      train/<breed>/*.jpg
      val/<breed>/*.jpg
      test/<breed>/*.jpg   # optional
