# Indian Cattle Breed Classifier

Identify Indian cattle breeds from a photo. Transfer learning in **PyTorch** on a
pretrained **timm** backbone (EfficientNet-B0 by default). Designed to train on
**Kaggle** (GPU), plus scripts for evaluation, single-image inference, and a
Gradio web demo.

## What's inside

```
cattle-breed-classifier/
├── config.yaml                 # all settings: data path, model, hyperparams
├── requirements.txt
├── notebooks/
│   └── kaggle_train.ipynb      # self-contained Kaggle training notebook
├── src/
│   ├── config.py               # YAML loader
│   ├── dataset.py              # transforms, stratified split, dataloaders
│   ├── model.py                # timm backbone + classifier head
│   ├── train.py                # training loop, checkpoints, freeze/unfreeze
│   ├── evaluate.py             # test accuracy, report, confusion matrix, curves
│   ├── inference.py            # predict breed for an image / folder (top-k)
│   └── utils.py                # seed, device, label map, meters
├── app/
│   └── app.py                  # Gradio upload-and-classify demo
└── scripts/
    └── prepare_data_example.md # expected data layout
```

## Data format

Folder-per-breed. Each breed gets its own sub-folder of images; the folder name
is the label. See `scripts/prepare_data_example.md`. No CSV required.


## Your dataset

This build is configured for a pre-split dataset: **64 Indian cattle/buffalo breeds**,
~12,325 train / 1,401 val / 879 test images (`.jpg/.png/.jpeg`), laid out as
`train/<breed>/`, `val/<breed>/`, `test/<breed>/`.

Note: `test/` has 63 breeds (missing `Holstein_Friesian`); the code locks all
splits to the train class order so labels stay aligned. On Kaggle, upload the
folder that contains `train/ val/ test/`, set `CONFIG["data_dir"]` to it, and keep
`has_split = True`.

## Train on Kaggle (recommended)

1. Create a new Kaggle Notebook and **upload `notebooks/kaggle_train.ipynb`**
   (File → Import Notebook), or copy its cells.
2. **Add your dataset** as input, then set `CONFIG["data_dir"]` in cell 2 to the
   breed folder, e.g. `/kaggle/input/indian-bovine-breeds/...`.
3. Settings → Accelerator → **GPU**.
4. **Run All.** Outputs (`best_model.pt`, `label_map.json`, plots, report) are
   written to `/kaggle/working/` — download them from the Output tab.

## Train locally

```bash
pip install -r requirements.txt
# edit config.yaml -> data.data_dir
python -m src.train --config config.yaml
python -m src.evaluate --config config.yaml
```

## Predict

```bash
python -m src.inference --image path/to/cow.jpg --topk 3
python -m src.inference --image path/to/folder --topk 3
```

## Web demo

```bash
python app/app.py                 # loads outputs/best_model.pt
python app/app.py --share         # public link
```

## Notes & tips

- **Backbone:** switch `model.backbone` in `config.yaml` to any timm model
  (`resnet50`, `convnext_tiny`, `vit_base_patch16_224`, `mobilenetv3_large_100`).
- **Freeze schedule:** the head trains alone for `freeze_epochs`, then the whole
  network unfreezes — stabilizes early training on small datasets.
- **Imbalanced breeds:** check `confusion_matrix.png`; add images for weak classes.
- **Checkpoint** stores the class list, backbone name, and image size, so
  inference/demo need no extra config.
