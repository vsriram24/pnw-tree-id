# PNW Tree Identifier

A deep learning application that identifies Pacific Northwest tree species from photos. Uses transfer learning with EfficientNetV2-S trained on iNaturalist research-grade observations, served via a Flask web app with drag-and-drop image upload.

## Species Coverage

Classifies **40 PNW tree species** across two groups:

**Conifers (22):** Douglas Fir, Western Red Cedar, Sitka Spruce, Western Hemlock, Mountain Hemlock, Ponderosa Pine, Lodgepole Pine, Western White Pine, Sugar Pine, Noble Fir, Grand Fir, Subalpine Fir, Pacific Silver Fir, White Fir, Incense Cedar, Alaska Yellow Cedar, Port Orford Cedar, Engelmann Spruce, Western Larch, Western Juniper, Pacific Yew, Coast Redwood

**Broadleaf (18):** Big Leaf Maple, Vine Maple, Red Alder, White Alder, Pacific Madrone, Oregon White Oak, Canyon Live Oak, Oregon Ash, Black Cottonwood, Quaking Aspen, Pacific Dogwood, Tanoak, California Laurel, Paper Birch, Golden Chinquapin, Cascara Buckthorn, Pacific Willow, Bitter Cherry

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download training data

Download research-grade observations from iNaturalist for all species:

```bash
python scripts/download_dataset.py
```

Or download specific species to test:

```bash
python scripts/download_dataset.py --species "Douglas Fir" "Red Alder"
```

### 2. Preprocess and split

Validates images, resizes to 384px, deduplicates, and creates a 70/15/15 stratified train/val/test split:

```bash
python scripts/prepare_dataset.py
```

### 3. Train

Two-phase training: first trains the classifier head with the backbone frozen, then fine-tunes the full model with differential learning rates:

```bash
python scripts/train.py
```

For a quick test run:

```bash
python scripts/train.py --phase1-epochs 2 --phase2-epochs 5 --batch-size 16
```

### 4. Evaluate

Run evaluation on the test set with per-class metrics and confusion matrix:

```bash
python scripts/evaluate.py
```

### 5. Web app

Start the Flask web app and open http://localhost:5000:

```bash
python webapp/app.py
```

Upload a tree photo via drag-and-drop to get top species predictions with confidence scores.

## Architecture

- **Backbone:** EfficientNetV2-S (ImageNet pretrained, 21.5M params)
- **Classifier head:** Dropout(0.3) &rarr; Linear(1280, 512) &rarr; ReLU &rarr; BatchNorm &rarr; Dropout(0.15) &rarr; Linear(512, 40)
- **Training:** AdamW optimizer, CrossEntropy with label smoothing (0.1), CosineAnnealing LR schedule, early stopping (patience 7)
- **Data:** ~400 images/species from iNaturalist, filtered to Oregon, Washington, and British Columbia

## Project Structure

```
pnw-tree-id/
├── config/species.yaml          # Species list with scientific names and taxon IDs
├── scripts/
│   ├── download_dataset.py      # Download images from iNaturalist
│   ├── prepare_dataset.py       # Preprocess and split dataset
│   ├── train.py                 # Train the model
│   └── evaluate.py              # Evaluate on test set
├── src/
│   ├── dataset/                 # Download, preprocess, split, transforms
│   ├── model/                   # Architecture and inference
│   └── training/                # Training loop and metrics
├── webapp/                      # Flask web app with drag-and-drop upload
├── data/                        # Downloaded and processed images (gitignored)
└── checkpoints/                 # Saved model weights (gitignored)
```
