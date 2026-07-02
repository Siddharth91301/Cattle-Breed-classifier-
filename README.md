Indian Cattle & Buffalo Breed Classifier

An image-classification project that identifies 64 Indian indigenous cattle and buffalo breeds
from a single photograph. It uses PyTorch transfer learning (EfficientNet-B3 via timm), a YOLO
cow-detector front-end for cropping and rejecting non-cattle images, and a small Flask web app for
inference. The project is motivated by national livestock-digitization efforts (Rashtriya Gokul
Mission, Bharat Pashudhan), where accurate on-device breed recognition can reduce data-entry errors
during animal registration.


Status: working classifier — best validation accuracy ~82%, test accuracy ~79% (macro-F1 ~0.75).
Actively improving rare-breed data quality. See Results and Known limitations.

Table of contents-
Overview
Dataset
Repository structure
Method
Results
Known limitations
How to train (Kaggle)
How to clean the data
How to run the web app
Future work



Overview

The task is single-label image classification — one breed per image — not object detection, so
no bounding-box annotation is required for training. Many Indian breeds are phenotypically similar
(subtle differences in coat colour, horn shape, hump, dewlap), photographed in uncontrolled field
conditions, which makes this a fine-grained recognition problem over a large (64-class) label space.


Dataset


64 breeds, pre-split into train/, val/, test/, each with one sub-folder per breed
(train/<breed>/*.jpg|jpeg|png).
Approx. 12,300 train / 1,400 val / ~870 test images.
test/ contains 63 breeds (missing Holstein_Friesian); all splits are locked to the
training class order so labels stay aligned.
Class imbalance: ~70–700 images per breed. Rare breeds are kept (not deleted) and addressed
via sampling and re-sourcing.



The dataset is uploaded to Kaggle as a versioned dataset; the training notebook is imported
separately. /kaggle/input is read-only — cleaning is done on a writable copy or locally.




Repository structure

File / folderPurposebreed_classifier_v2.ipynbMain training notebook (data loading, model, train, eval, plots).process_data.pyOffline dataset top-up with quality gate (min-resolution, junk-filename, dedup).download_images.pyAuto-download candidate breed photos from image search into _incoming/.intake_images.pyValidate (>=400px), pHash-dedupe, and split new images 80/10/10 into the dataset.quarantine_junk.pyNon-destructively move thumbnails/junk to _review/ (reversible via manifest).breed_app/Flask web app for inference (YOLO crop + top-3 + confidence gate).data_cleaning_prompt.mdStandalone brief for continuing the data-cleaning work.project_context_prompt.mdFull project context for resuming in a new session._review/, _incoming/Working folders (quarantine, download staging). Not part of the dataset.


Method

Transfer learning, two-phase. An ImageNet-pretrained tf_efficientnet_b3.ns_jft_in1k
(NoisyStudent) backbone with a new 64-way head. The backbone is frozen for 3 epochs (head-only),
then unfrozen for end-to-end fine-tuning with cosine LR decay. AdamW, mixed precision, gradient
clipping, early stopping on validation accuracy.

Anti-overfitting recipe (v2):


Input resolution 300x300 (B3 native).
Dropout 0.4, weight decay 0.01.
RandAugment + Random Erasing.
Class imbalance via a WeightedRandomSampler (sqrt inverse-frequency), which composes with MixUp.
Test-time augmentation (image + horizontal flip).
MixUp / CutMix with soft-target cross-entropy (strongest regularizer).


Evaluation: overall accuracy, macro-F1 (weights every breed equally, reflecting rare-breed
performance), per-breed classification report, normalized confusion matrix, and most-confused pairs.

Inference (web app): YOLO detects the largest cow -> crop -> classify the crop -> softmax top-3.
Images with no detected animal are rejected; low-confidence top-1 is flagged rather than asserted.


Results

VersionBackbone / epochsTest accMacro-F1Notesv1EfficientNet-B3, 2576.8%0.74Heavy overfitting (train 98.6%).v2.0EfficientNet-B3, 3077.6%0.749Overfitting fixed but undertrained.v2.1NoisyStudent-B3, 45~79%~0.75Best val ~82%; converged.

The regularization closed the train/val gap; longer training + NoisyStudent weights lifted accuracy.
Well-represented breeds are strong (Sahiwal, Jaffrabadi, Kankrej, Nili_Ravi all F1 >= 0.9).


Note on the test number: the test set currently contains some mislabeled non-cattle images in
the rare-breed folders, which understates true accuracy. Cleaning val/ and test/ (see below) is
required before the reported number can be fully trusted.




Known limitations


~7 rare breeds (Bachaur, Shweta Kapila, Gangatari, Dagri, Badri, Nari, Kherigarh) have low
F1 (~0.1-0.3). Their web-sourced images are polluted with non-cattle content (product photos,
breed charts, cartoons, landscapes) that filename/size/OCR filters miss. The fix is a cow-detector
cleaning pass (drop any image with no cow) plus manual review — see the cleaning brief.
Real-world vs test accuracy: the test set resembles training (clean, posed web photos). Field
photos from phones (odd angles, clutter, distance) will score lower; the YOLO crop front-end
mitigates this but does not eliminate it.
The classifier has no built-in "unknown" class; out-of-distribution rejection is handled at
inference by the detector + confidence gate.



How to train (Kaggle)


Upload the dataset (as a new version of the existing Kaggle dataset — keep only train/ val/ test/).
Open breed_classifier_v2.ipynb as a Kaggle notebook, attach the dataset, and enable
GPU (T4) and Internet (for pretrained weights).
Run all cells. The notebook auto-detects the data path, trains, evaluates with TTA, and writes
best_model.pt, label_map.json, history.json, and plots to /kaggle/working.
Download those outputs before closing the session (or use Save Version -> Save & Run All),
or they are lost.



How to clean the data

The reliable cleaner is a pretrained cow detector: drop any image where no cow is found (catches
product shots, charts, landscapes, cartoons that text/size filters miss), then optionally crop to the
animal. Run it across all three splits (train/val/test), non-destructively, then do a short manual
pass on the survivors of the rare breeds.

See data_cleaning_prompt.md for a complete, standalone brief you can hand to a fresh session.
Existing helpers: quarantine_junk.py (size/filename gate, reversible), download_images.py +
intake_images.py (re-source + validate + split), process_data.py (gated top-up).


Always eyeball downloaded images before ingesting them — automated search returns wrong animals,
ads, and infographics that will poison the labels if added blindly.




How to run the web app

The Flask app lives in breed_app/ with best_model.pt and label_map.json already inside.

bashcd breed_app
python -m venv venv
venv\Scripts\activate          # Windows   (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python app.py

Open http://127.0.0.1:5000 . Upload a clear side-on photo of a single animal; the page shows the
top-3 breeds with confidence, or "No cow detected" / "Not confident" when appropriate. The first run
downloads the small YOLO weights (needs internet once). Tune the confidence gate with
CONF_THRESHOLD=0.55 python app.py. See breed_app/README.md for details.


Future work


Finish the detector-based cleaning of train/val/test, then retrain.
Crop-to-animal at training time; raise input resolution to 384.
Try ConvNeXt / EfficientNetV2 backbones; ensemble for a final accuracy push.
Decoupled classifier re-training (cRT) to lift rare-breed F1.
Deploy as an assistive tool in the livestock-registration workflow (top-3, human-confirmed).



Acknowledgements

Breed data and references draw on ICAR-NBAGR breed descriptors and published Indian bovine
breed-recognition literature. This is a research/education project and an assistive tool — breed
predictions should be confirmed by a domain expert before being recorded.
