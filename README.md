# PNW Tree Identifier

**A deep learning-powered field guide for the forests of the Pacific Northwest.**

Snap a photo of a tree, and this app will tell you what species it is -- from the towering Douglas Firs of the Cascades to the papery-barked Madrones along the coast. Built with transfer learning on ~15,000 community-contributed observations from [iNaturalist](https://www.inaturalist.org/), it classifies **40 native PNW tree species** through a Flask web app with drag-and-drop simplicity.

---

## What's Inside

| | |
|---|---|
| **40 species** | Conifers + broadleaf trees native to OR, WA, and BC |
| **~15,500 images** | Research-grade photos from iNaturalist citizen scientists |
| **EfficientNetV2-S** | State-of-the-art image classifier with transfer learning |
| **Two-phase training** | Frozen-backbone warmup, then full fine-tuning |
| **Flask web app** | Drag-and-drop interface with instant predictions |

---

## Species Coverage

### Conifers (22 species)

| Common Name | Scientific Name | Notable Traits |
|---|---|---|
| Douglas Fir | *Pseudotsuga menziesii* | PNW's iconic timber tree; "mouse tail" bracts on cones |
| Western Red Cedar | *Thuja plicata* | Fibrous bark, drooping branches; culturally vital to Indigenous peoples |
| Sitka Spruce | *Picea sitchensis* | World's largest spruce; sharp, square needles |
| Western Hemlock | *Tsuga heterophylla* | WA state tree; graceful drooping leader |
| Mountain Hemlock | *Tsuga mertensiana* | Subalpine specialist; star-shaped needle arrangement |
| Ponderosa Pine | *Pinus ponderosa* | Puzzle-piece bark that smells like vanilla or butterscotch |
| Lodgepole Pine | *Pinus contorta* | Serotinous cones that open after fire |
| Western White Pine | *Pinus monticola* | ID state tree; long, slender cones |
| Sugar Pine | *Pinus lambertiana* | World's longest pine cones (up to 60 cm!) |
| Noble Fir | *Abies procera* | Tallest true fir; popular Christmas tree |
| Grand Fir | *Abies grandis* | Flat, glossy needles in two ranks; citrus scent |
| Subalpine Fir | *Abies lasiocarpa* | Narrow spire shape; high-elevation forests |
| Pacific Silver Fir | *Abies amabilis* | Silvery-white undersides on needles |
| White Fir | *Abies concolor* | Blue-green needles; common at mid elevations |
| Incense Cedar | *Calocedrus decurrens* | Classic "pencil cedar"; cinnamon-red bark |
| Alaska Yellow Cedar | *Callitropsis nootkatensis* | Weeping form; extremely rot-resistant wood |
| Port Orford Cedar | *Chamaecyparis lawsoniana* | Limited range in SW Oregon; highly valued timber |
| Engelmann Spruce | *Picea engelmannii* | High-elevation spruce; papery, scaly bark |
| Western Larch | *Larix occidentalis* | A deciduous conifer -- drops its needles in fall! |
| Western Juniper | *Juniperus occidentalis* | Gnarled, drought-adapted; east-side PNW |
| Pacific Yew | *Taxus brevifolia* | Source of taxol (cancer treatment); red berry-like arils |
| Coast Redwood | *Sequoia sempervirens* | Earth's tallest tree; southern OR populations |

### Broadleaf Trees (18 species)

| Common Name | Scientific Name | Notable Traits |
|---|---|---|
| Big Leaf Maple | *Acer macrophyllum* | Largest maple leaves in North America (30 cm+) |
| Vine Maple | *Acer circinatum* | Multi-stemmed; spectacular fall color |
| Red Alder | *Alnus rubra* | Nitrogen-fixer; smooth grey bark with lichens |
| White Alder | *Alnus rhombifolia* | Streamside specialist; southern range |
| Pacific Madrone | *Arbutus menziesii* | Peeling cinnamon bark; evergreen broadleaf |
| Oregon White Oak | *Quercus garryana* | Gnarled silhouette; savanna ecosystem anchor |
| Canyon Live Oak | *Quercus chrysolepis* | Golden-fuzzy acorn cups; rocky slopes |
| Oregon Ash | *Fraxinus latifolia* | Only native PNW ash; wetland indicator |
| Black Cottonwood | *Populus trichocarpa* | Tallest North American broadleaf; cotton-like seeds |
| Quaking Aspen | *Populus tremuloides* | Trembling leaves; massive clonal colonies |
| Pacific Dogwood | *Cornus nuttallii* | Showy white bracts; BC provincial flower |
| Tanoak | *Notholithocarpus densiflorus* | Not a true oak despite acorn-like nuts |
| California Laurel | *Umbellularia californica* | Intensely aromatic crushed leaves |
| Paper Birch | *Betula papyrifera* | Classic white, peeling bark |
| Golden Chinquapin | *Chrysolepis chrysophylla* | Golden leaf undersides; spiny burr fruits |
| Cascara Buckthorn | *Frangula purshiana* | Bark used traditionally as a laxative |
| Pacific Willow | *Salix lasiandra* | Glossy lance-shaped leaves; riparian zones |
| Bitter Cherry | *Prunus emarginata* | Shiny reddish bark; clusters of white flowers |

---

## How It Works

The system follows a four-stage pipeline: data collection, preprocessing, model training, and inference. Here's the full picture:

```
                            PNW Tree Identifier Pipeline
 ============================================================================

 1. DATA COLLECTION             2. PREPROCESSING
 ----------------------         ----------------------
 iNaturalist API v1             Validate & clean
   |                              |
   v                              v
 ~400 research-grade            Resize (384px long edge)
 photos per species               |
   |                              v
   v                            SHA256 deduplicate
 40 species x 3 regions          |
 (OR, WA, BC)                    v
   |                            Stratified split
   v                            70% / 15% / 15%
 ~15,500 total images           (train / val / test)


 3. MODEL TRAINING              4. INFERENCE
 ----------------------         ----------------------
 EfficientNetV2-S               Upload photo (web app)
 (ImageNet pretrained)            |
   |                              v
   v                            Resize + CenterCrop 384px
 Phase 1: Freeze backbone        |
 Train classifier head            v
 (5 epochs, lr=1e-3)           EfficientNetV2-S forward pass
   |                              |
   v                              v
 Phase 2: Unfreeze all          Softmax -> top-k predictions
 Differential LR                  |
 (backbone=1e-5, head=1e-4)      v
 Cosine annealing               "{species}: {confidence}%"
 Early stopping (patience=7)
```

---

## Data: Sourced from the Community

### Why iNaturalist?

[iNaturalist](https://www.inaturalist.org/) is a citizen science platform where naturalists worldwide upload observations of living organisms. Each observation includes photos, a GPS location, a date, and a community-verified species identification. Observations that reach "research grade" have been confirmed by multiple independent identifiers, making them a reliable (and free!) source of labeled training data.

### How We Collected It

For each of the 40 species, the download pipeline:

1. **Queries the iNaturalist API** (`/v1/observations`) filtering by:
   - **Taxon ID** -- each species' unique identifier (e.g., Douglas Fir = taxon 48256)
   - **Place IDs** -- Oregon (10), Washington (46), British Columbia (7085)
   - **Quality grade** -- `research` only (community-verified identifications)
   - **Has photos** -- observations must include at least one image

2. **Extracts photo URLs** from each observation, upgrading thumbnails from `square` to `medium` resolution (~500px) for better training quality.

3. **Downloads with rate limiting** (1 request/second) to respect iNaturalist's API guidelines. The pipeline supports resume -- if interrupted, it picks up where it left off.

4. **Deduplicates** by observation ID to avoid counting multiple photos from the same sighting.

The result: roughly **400 images per species** (~15,500 total), representing the full diversity of how each tree appears across seasons, lighting conditions, angles, and growth stages.

### Preprocessing

Before training, every image passes through a quality pipeline:

- **Validation:** Corrupt or un-openable images are discarded
- **Resizing:** Longest edge scaled to 384px (preserving aspect ratio) to standardize input size without distortion
- **Deduplication:** SHA256 hashes catch exact-duplicate images that may appear across different observations
- **RGB conversion:** All images converted to 3-channel RGB at JPEG quality 95
- **Stratified splitting:** scikit-learn's `train_test_split` creates a 70/15/15 train/val/test split, maintaining class proportions across all three sets

---

## Model Architecture

### Why EfficientNetV2-S?

[EfficientNetV2](https://arxiv.org/abs/2104.00298) is a family of image classifiers designed by Google using neural architecture search (NAS). The "S" (small) variant strikes a balance between accuracy and computational cost -- 21.5 million parameters, fast enough to run on a laptop, accurate enough for fine-grained species classification.

We use a model **pretrained on ImageNet** (1.28 million images, 1000 classes). ImageNet includes plenty of natural scenes, textures, and shapes, so the backbone already "knows" useful visual features like bark patterns, leaf shapes, and branching structures. We just need to teach the final layers what PNW trees look like.

### Architecture Diagram

```
 Input Image (384 x 384 x 3)
          |
          v
 +------------------------------+
 |    EfficientNetV2-S           |
 |    Backbone (Features)        |
 |                                |
 |  - Fused-MBConv blocks        |
 |  - MBConv blocks              |
 |  - Progressive learning       |
 |  - 21.5M parameters           |
 |  - ImageNet pretrained         |
 +------------------------------+
          |
          v
  Feature Vector (1280-dim)
          |
          v
 +------------------------------+
 |    Custom Classifier Head     |
 |                                |
 |  Dropout(0.3)                 |
 |       |                       |
 |  Linear(1280 -> 512)         |
 |       |                       |
 |  ReLU + BatchNorm(512)       |
 |       |                       |
 |  Dropout(0.15)               |
 |       |                       |
 |  Linear(512 -> 40 classes)   |
 +------------------------------+
          |
          v
  Softmax -> Predicted Species
```

### Transfer Learning: Two-Phase Training

Rather than training from scratch, we use **transfer learning** in two phases:

#### Phase 1: Train the Classifier Head (5 epochs)

The backbone is **frozen** (no gradient updates). Only the new classifier head learns, using a relatively high learning rate (`1e-3`). This is like installing a new "tree expert" brain on top of the existing visual system -- we teach it to map ImageNet features to our 40 species without disturbing the backbone's learned representations.

#### Phase 2: Fine-tune Everything (up to 30 epochs)

Now we **unfreeze the backbone** and train the entire network with **differential learning rates**:
- **Backbone:** `1e-5` (gentle updates -- these features are already good)
- **Classifier head:** `1e-4` (10x higher -- still adapting to our specific task)

This allows the backbone to subtly adjust its feature extraction for tree-specific patterns (e.g., bark texture, needle vs. leaf distinctions) while the classifier continues refining its decision boundaries.

#### Training Techniques

| Technique | What It Does | Why It Helps |
|---|---|---|
| **Label smoothing (0.1)** | Softens one-hot targets to 90% correct / ~0.26% each other class | Prevents overconfidence; improves calibration |
| **Cosine annealing LR** | Learning rate follows a cosine curve from initial to near-zero | Smooth convergence; avoids sharp LR drops |
| **Early stopping (patience 7)** | Stops if validation loss doesn't improve for 7 epochs | Prevents overfitting; saves compute |
| **AdamW optimizer** | Adam with decoupled weight decay (0.01) | Better generalization than vanilla Adam |
| **Data augmentation** | Random crops, flips, color jitter, rotation, erasing | Teaches invariance to viewpoint, lighting, occlusion |

### Data Augmentation Details

Training images are transformed on-the-fly to increase effective dataset size and teach the model to be robust:

| Transform | Parameters | Purpose |
|---|---|---|
| RandomResizedCrop | 384px, scale 0.6-1.0 | Simulates different distances and framings |
| RandomHorizontalFlip | p=0.5 | Trees look the same flipped |
| ColorJitter | brightness/contrast/saturation 0.3, hue 0.05 | Handles varying light and seasons |
| RandomRotation | +/- 15 degrees | Compensates for tilted camera angles |
| RandomErasing | p=0.2, scale 0.02-0.15 | Simulates occlusion (branches, signs, etc.) |
| ImageNet normalization | mean=[0.485, 0.456, 0.406] | Matches the pretrained backbone's expectations |

At **inference time**, augmentation is disabled -- images are deterministically resized to 422px then center-cropped to 384px for consistent predictions.

---

## Web Application

The Flask web app provides a simple drag-and-drop interface:

1. **Upload** a photo (JPG, PNG, or WebP up to 16MB)
2. **Preview** your image before submitting
3. **Identify** -- the model returns the top-5 most likely species with confidence scores
4. **Learn** -- an "About" section explains the project's background, data, and methodology

Under the hood:
- Images are saved temporarily, validated as real images via PIL, then passed to the model
- The `TreeClassifier` singleton loads the checkpoint once and stays in memory
- Predictions use softmax over 40 output logits to produce calibrated confidence scores
- Uploaded files are cleaned up immediately after prediction

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the web app
python webapp/app.py

# Open http://localhost:5000
```

---

## Full Pipeline: From Zero to Predictions

### 1. Download training data

```bash
# All 40 species (~15,500 images, takes a while due to rate limiting)
python scripts/download_dataset.py

# Or just a few species to test
python scripts/download_dataset.py --species "Douglas Fir" "Red Alder"
```

### 2. Preprocess and split

```bash
# Validates, resizes, deduplicates, and creates 70/15/15 split
python scripts/prepare_dataset.py
```

### 3. Train the model

```bash
# Full training (two-phase, ~35 epochs)
python scripts/train.py

# Quick test run
python scripts/train.py --phase1-epochs 2 --phase2-epochs 5 --batch-size 16
```

### 4. Evaluate

```bash
# Per-class metrics, confusion matrix, top-1/top-5 accuracy
python scripts/evaluate.py
```

### 5. Launch the web app

```bash
python webapp/app.py
# Visit http://localhost:5000
```

---

## Project Structure

```
pnw-tree-id/
├── config/
│   └── species.yaml              # 40 species: names, taxon IDs, regions
├── scripts/
│   ├── download_dataset.py       # Fetch images from iNaturalist API
│   ├── prepare_dataset.py        # Validate, resize, deduplicate, split
│   ├── train.py                  # Two-phase training loop
│   └── evaluate.py               # Test-set metrics + confusion matrix
├── src/
│   ├── dataset/
│   │   ├── download.py           # iNaturalist API integration
│   │   ├── preprocess.py         # Image validation & cleaning
│   │   ├── split.py              # Stratified train/val/test split
│   │   └── transforms.py         # Augmentation & normalization
│   ├── model/
│   │   ├── architecture.py       # EfficientNetV2-S + custom head
│   │   └── inference.py          # TreeClassifier prediction wrapper
│   └── training/
│       ├── trainer.py            # Training loop & checkpointing
│       └── metrics.py            # Accuracy, F1, confusion matrix
├── webapp/
│   ├── app.py                    # Flask application factory
│   ├── routes.py                 # / and /predict endpoints
│   ├── model_loader.py           # Singleton model loader
│   ├── templates/
│   │   ├── base.html             # Base template with nav and footer
│   │   ├── index.html            # Upload interface
│   │   └── about.html            # Methodology deep-dive
│   └── static/
│       ├── css/style.css         # Forest-themed responsive styles
│       └── js/upload.js          # Drag-and-drop + prediction display
├── data/                         # Downloaded & processed images (gitignored)
├── checkpoints/                  # Model weights (gitignored)
├── requirements.txt
└── README.md                     # You are here
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep learning | PyTorch 2.0+, torchvision |
| Model | EfficientNetV2-S (via torchvision) |
| Data source | iNaturalist API v1 |
| Image processing | Pillow, torchvision transforms |
| ML utilities | scikit-learn, matplotlib, seaborn |
| Web framework | Flask 3.0+ |
| Config | PyYAML |

---

## Acknowledgments

- **iNaturalist** and its global community of citizen scientists for providing the research-grade observations that make this project possible. Every photo in the training set was contributed and verified by real people exploring nature.
- **ImageNet** and the researchers behind [EfficientNetV2](https://arxiv.org/abs/2104.00298) for the pretrained backbone.
- The forests of the Pacific Northwest for being endlessly beautiful and worth learning about.
