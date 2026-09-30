# ================================================================
# HERITAGE AI
# HADAR + EWAR RESEARCH PROTOTYPE
# ================================================================
#
# Dataset:
# /content/all.zip
#
# Current classes:
#   gopuram     -> Gopuram
#   pillars     -> Pillar
#   mandapam    -> Mandapa
#   sculptures  -> Sculpture
#   vimana      -> Vimana (optional if added later)
#
# Main research pipeline:
#
# Raw Images
#      ↓
# Image Validation & Cleaning
#      ↓
# Duplicate Detection
#      ↓
# Train / Validation / Test
#      ↓
# EfficientNet-B0 Transfer Learning
#      ↓
# Architectural Feature Embedding
#      ↓
# Architectural DNA
#      ↓
# Heritage Knowledge Graph
#      ↓
# Evidence Retrieval
#      ↓
# EWAR Evidence Score
#      ↓
# XAI Grad-CAM
#
# IMPORTANT:
# EWAR is a proposed research framework/prototype.
# It must be experimentally validated before claiming a new algorithm.
#
# The reconstruction component is reference-based evidence retrieval.
# It does NOT claim historical authenticity.
# ================================================================


# ================================================================
# 1. INSTALL LIBRARIES
# ================================================================

!pip -q install torch torchvision
!pip -q install pandas numpy matplotlib seaborn pillow imagehash networkx scikit-learn tqdm


# ================================================================
# 2. IMPORT LIBRARIES
# ================================================================

import os
import zipfile
import shutil
import hashlib
import json
import random
import warnings

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm.auto import tqdm

import imagehash
import networkx as nx

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torchvision.models import EfficientNet_B0_Weights


warnings.filterwarnings("ignore")


# ================================================================
# 3. REPRODUCIBILITY
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ================================================================
# 4. DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("HERITAGE AI")
print("=" * 60)

print("Device:", DEVICE)


# ================================================================
# 5. PATHS
# ================================================================

ZIP_PATH = Path("/content/all.zip")

BASE_DIR = Path("/content")

RAW_DIR = BASE_DIR / "heritage_raw"

CLEAN_DIR = BASE_DIR / "heritage_clean"

SPLIT_DIR = BASE_DIR / "heritage_split"

OUTPUT_DIR = BASE_DIR / "heritage_outputs"

MODEL_PATH = BASE_DIR / "heritage_ai_model.pth"

METADATA_PATH = BASE_DIR / "heritage_metadata.csv"


# ================================================================
# 6. SETTINGS
# ================================================================

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 15

LEARNING_RATE = 0.0001

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


# ================================================================
# 7. CHECK ZIP
# ================================================================

if not ZIP_PATH.exists():

    raise FileNotFoundError(
        """
        /content/all.zip was not found.

        Upload all.zip to Colab and run the cell again.
        """
    )

print("\nZIP FOUND")

zip_size = ZIP_PATH.stat().st_size / (1024 * 1024)

print(
    "ZIP size:",
    round(zip_size, 2),
    "MB"
)


# ================================================================
# 8. EXTRACT DATASET
# ================================================================

if RAW_DIR.exists():

    shutil.rmtree(RAW_DIR)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


with zipfile.ZipFile(
    ZIP_PATH,
    "r"
) as zip_ref:

    zip_ref.extractall(
        RAW_DIR
    )


print(
    "\nDataset extracted to:",
    RAW_DIR
)


# ================================================================
# 9. DETECT CLASS FOLDERS
# ================================================================

# Your actual folder names are:
#
# gopuram
# pillars
# mandapam
# sculptures
#
# We map them into research names.

CLASS_MAPPING = {

    "gopuram": "Gopuram",

    "pillar": "Pillar",

    "pillars": "Pillar",

    "mandapa": "Mandapa",

    "mandapam": "Mandapa",

    "sculpture": "Sculpture",

    "sculptures": "Sculpture",

    "vimana": "Vimana"

}


# Find folders recursively

candidate_folders = []

for folder in RAW_DIR.rglob("*"):

    if not folder.is_dir():
        continue

    folder_name = folder.name.lower()

    if folder_name in CLASS_MAPPING:

        candidate_folders.append(folder)


if len(candidate_folders) == 0:

    raise RuntimeError(
        "No recognized class folders were found."
    )


print("\nRecognized folders:")

for folder in candidate_folders:

    print(
        folder,
        "→",
        CLASS_MAPPING[folder.name.lower()]
    )


# ================================================================
# 10. CREATE CLEAN DATASET DIRECTORY
# ================================================================

if CLEAN_DIR.exists():

    shutil.rmtree(CLEAN_DIR)


CLEAN_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# 11. IMAGE HASH FUNCTIONS
# ================================================================

def sha256_hash(file_path):

    h = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            h.update(chunk)

    return h.hexdigest()


def perceptual_hash(file_path):

    try:

        with Image.open(
            file_path
        ) as img:

            return str(
                imagehash.phash(
                    img.convert("RGB")
                )
            )

    except:

        return None


# ================================================================
# 12. CLEANING
# ================================================================

metadata_records = []

exact_hashes = set()

perceptual_hashes = {}

global_index = 1


print("\n")
print("=" * 60)
print("IMAGE CLEANING")
print("=" * 60)


for source_folder in sorted(
    candidate_folders
):

    original_class = source_folder.name.lower()

    class_name = CLASS_MAPPING[
        original_class
    ]

    destination_folder = (
        CLEAN_DIR / class_name
    )

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    class_counter = 1


    files = [

        f

        for f in source_folder.rglob("*")

        if f.is_file()
        and f.suffix.lower()
        in VALID_EXTENSIONS

    ]


    for image_path in tqdm(
        files,
        desc=class_name
    ):

        try:

            # ------------------------------------------------
            # Validate image
            # ------------------------------------------------

            with Image.open(
                image_path
            ) as img:

                img = img.convert(
                    "RGB"
                )

                width, height = img.size

                img.load()


            # ------------------------------------------------
            # Reject very small images
            # ------------------------------------------------

            if width < 80 or height < 80:

                continue


            # ------------------------------------------------
            # Exact duplicate detection
            # ------------------------------------------------

            file_hash = sha256_hash(
                image_path
            )

            if file_hash in exact_hashes:

                print(
                    "Exact duplicate:",
                    image_path.name
                )

                continue


            exact_hashes.add(
                file_hash
            )


            # ------------------------------------------------
            # Perceptual hash
            # ------------------------------------------------

            phash = perceptual_hash(
                image_path
            )

            near_duplicate = False

            if phash is not None:

                for existing_hash, existing_info in perceptual_hashes.items():

                    try:

                        distance = (
                            imagehash.hex_to_hash(phash)
                            - imagehash.hex_to_hash(existing_hash)
                        )

                        if distance <= 3:

                            near_duplicate = True

                            print(
                                "Near duplicate candidate:",
                                image_path.name
                            )

                            break

                    except:

                        pass


            # We DO NOT automatically delete near duplicates.
            # They are recorded so that manual visual inspection
            # can make the final decision.

            # ------------------------------------------------
            # Standardized name
            # ------------------------------------------------

            new_name = (
                f"{class_name}_{class_counter:03d}.jpg"
            )

            destination_path = (
                destination_folder /
                new_name
            )


            # ------------------------------------------------
            # Save standardized JPG
            # ------------------------------------------------

            with Image.open(
                image_path
            ) as img:

                img = img.convert(
                    "RGB"
                )

                img.save(
                    destination_path,
                    "JPEG",
                    quality=95
                )


            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata_records.append({

                "image_id":
                    new_name,

                "class":
                    class_name,

                "original_class":
                    original_class,

                "original_file":
                    str(image_path),

                "clean_file":
                    str(destination_path),

                "width":
                    width,

                "height":
                    height,

                "aspect_ratio":
                    round(
                        width / height,
                        4
                    ),

                "sha256":
                    file_hash,

                "phash":
                    phash,

                "near_duplicate_candidate":
                    near_duplicate,

                "source":
                    "Google Images",

                "license_status":
                    "Unverified"

            })


            if phash is not None:

                perceptual_hashes[
                    phash
                ] = {

                    "file":
                        str(image_path),

                    "class":
                        class_name

                }


            class_counter += 1

            global_index += 1


        except Exception as e:

            print(
                "Invalid image:",
                image_path
            )

            print(
                "Reason:",
                e
            )


# ================================================================
# 13. SAVE METADATA
# ================================================================

metadata = pd.DataFrame(
    metadata_records
)

metadata.to_csv(
    METADATA_PATH,
    index=False
)


print("\n")
print("=" * 60)
print("CLEANING RESULT")
print("=" * 60)

print(
    "Total images:",
    len(metadata)
)


print("\nClass distribution:")

print(
    metadata["class"]
    .value_counts()
)


print(
    "\nMetadata saved:",
    METADATA_PATH
)


# ================================================================
# 14. VISUAL CONTACT SHEETS
# ================================================================

CONTACT_DIR = (
    OUTPUT_DIR /
    "contact_sheets"
)

CONTACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def create_contact_sheet(
    class_name,
    max_images=40
):

    folder = (
        CLEAN_DIR /
        class_name
    )

    images = sorted(
        folder.glob("*.jpg")
    )[:max_images]


    if len(images) == 0:

        return


    columns = 5

    rows = int(
        np.ceil(
            len(images) /
            columns
        )
    )


    fig, axes = plt.subplots(

        rows,
        columns,

        figsize=(
            16,
            3 * rows
        )

    )


    axes = np.array(
        axes
    ).reshape(-1)


    for i, image_path in enumerate(
        images
    ):

        try:

            img = Image.open(
                image_path
            ).convert("RGB")

            axes[i].imshow(
                img
            )

            axes[i].set_title(
                image_path.name,
                fontsize=8
            )

            axes[i].axis("off")

        except:

            axes[i].axis("off")


    for j in range(
        len(images),
        len(axes)
    ):

        axes[j].axis("off")


    fig.suptitle(
        f"{class_name} - Visual Cleaning Sheet",
        fontsize=16
    )


    plt.tight_layout()

    output_file = (
        CONTACT_DIR /
        f"{class_name}_contact_sheet.jpg"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


print("\nCreating visual inspection sheets...")

for class_name in sorted(
    metadata["class"].unique()
):

    create_contact_sheet(
        class_name
    )


print(
    "\nContact sheets:",
    CONTACT_DIR
)


# ================================================================
# 15. DATASET DISTRIBUTION PLOT
# ================================================================

plt.figure(
    figsize=(8,5)
)

metadata["class"].value_counts().plot(
    kind="bar"
)

plt.title(
    "HERITAGE AI Dataset Distribution"
)

plt.xlabel(
    "Architectural Element"
)

plt.ylabel(
    "Number of Images"
)

plt.tight_layout()

plt.show()


# ================================================================
# 16. TRAIN / VALIDATION / TEST SPLIT
# ================================================================

if SPLIT_DIR.exists():

    shutil.rmtree(
        SPLIT_DIR
    )


for split in [
    "train",
    "validation",
    "test"
]:

    for class_name in sorted(
        metadata["class"].unique()
    ):

        (
            SPLIT_DIR /
            split /
            class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


split_records = []


for class_name in sorted(
    metadata["class"].unique()
):

    class_data = metadata[
        metadata["class"] ==
        class_name
    ].sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)


    n = len(class_data)


    test_count = max(
        1,
        round(
            n * 0.15
        )
    )

    validation_count = max(
        1,
        round(
            n * 0.15
        )
    )


    test_data = class_data[
        :test_count
    ]

    validation_data = class_data[
        test_count:
        test_count + validation_count
    ]

    train_data = class_data[
        test_count + validation_count:
    ]


    for split_name, split_data in [

        ("train", train_data),

        ("validation", validation_data),

        ("test", test_data)

    ]:

        for _, row in split_data.iterrows():

            source = Path(
                row["clean_file"]
            )

            destination = (
                SPLIT_DIR /
                split_name /
                class_name /
                source.name
            )


            shutil.copy2(
                source,
                destination
            )


            row_dict = row.to_dict()

            row_dict["split"] = (
                split_name
            )

            split_records.append(
                row_dict
            )


split_metadata = pd.DataFrame(
    split_records
)

split_metadata.to_csv(
    BASE_DIR /
    "heritage_split_metadata.csv",
    index=False
)


print("\n")
print("=" * 60)
print("DATASET SPLIT")
print("=" * 60)


for split_name in [
    "train",
    "validation",
    "test"
]:

    count = sum(
        1
        for f
        in (
            SPLIT_DIR /
            split_name
        ).rglob("*")
        if f.is_file()
    )

    print(
        split_name,
        ":",
        count
    )


# ================================================================
# 17. PREPROCESSING / TRANSFORMS
# ================================================================

weights = (
    EfficientNet_B0_Weights.DEFAULT
)


train_transforms = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=8
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=weights.transforms().mean,

        std=weights.transforms().std

    )

])


evaluation_transforms = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=weights.transforms().mean,

        std=weights.transforms().std

    )

])


# ================================================================
# 18. IMAGEFOLDER DATASETS
# ================================================================

train_dataset = datasets.ImageFolder(

    SPLIT_DIR /
    "train",

    transform=train_transforms

)


validation_dataset = datasets.ImageFolder(

    SPLIT_DIR /
    "validation",

    transform=evaluation_transforms

)


test_dataset = datasets.ImageFolder(

    SPLIT_DIR /
    "test",

    transform=evaluation_transforms

)


CLASS_NAMES = train_dataset.classes

NUM_CLASSES = len(
    CLASS_NAMES
)


print("\n")
print("=" * 60)
print("MODEL CLASSES")
print("=" * 60)

print(
    CLASS_NAMES
)


# ================================================================
# 19. DATA LOADERS
# ================================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=2,

    pin_memory=torch.cuda.is_available()

)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2,

    pin_memory=torch.cuda.is_available()

)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2,

    pin_memory=torch.cuda.is_available()

)


# ================================================================
# 20. EFFICIENTNET-B0 TRANSFER LEARNING
# ================================================================

print("\n")
print("=" * 60)
print("LOADING EFFICIENTNET-B0")
print("=" * 60)


model = models.efficientnet_b0(
    weights=weights
)


# Freeze most feature layers

for parameter in model.features.parameters():

    parameter.requires_grad = False


# Unfreeze final feature blocks

for block in list(
    model.features.children()
)[-2:]:

    for parameter in block.parameters():

        parameter.requires_grad = True


# Replace classification head

input_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(

    input_features,

    NUM_CLASSES

)


model = model.to(
    DEVICE
)


# ================================================================
# 21. LOSS + OPTIMIZER
# ================================================================

criterion = (
    nn.CrossEntropyLoss()
)


optimizer = torch.optim.AdamW(

    filter(
        lambda p:
        p.requires_grad,

        model.parameters()
    ),

    lr=LEARNING_RATE,

    weight_decay=1e-4

)


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )
)


# ================================================================
# 22. TRAINING FUNCTION
# ================================================================

def run_epoch(
    loader,
    training=True
):

    model.train(
        training
    )


    total_loss = 0.0

    correct = 0

    total = 0


    for images, labels in loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        if training:

            optimizer.zero_grad()


        with torch.set_grad_enabled(
            training
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )


            if training:

                loss.backward()

                optimizer.step()


        total_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = (
            outputs.argmax(
                dim=1
            )
        )


        correct += (
            predictions ==
            labels
        ).sum().item()


        total += (
            labels.size(0)
        )


    return (

        total_loss / total,

        correct / total

    )


# ================================================================
# 23. TRAIN MODEL
# ================================================================

print("\n")
print("=" * 60)
print("TRAINING")
print("=" * 60)


best_validation_accuracy = 0.0

history = []


for epoch in range(
    EPOCHS
):

    train_loss, train_accuracy = (
        run_epoch(
            train_loader,
            training=True
        )
    )


    validation_loss, validation_accuracy = (
        run_epoch(
            validation_loader,
            training=False
        )
    )


    scheduler.step(
        validation_accuracy
    )


    history.append({

        "epoch":
            epoch + 1,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "validation_loss":
            validation_loss,

        "validation_accuracy":
            validation_accuracy

    })


    print(

        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.3f} | "
        f"Val Loss: {validation_loss:.4f} | "
        f"Val Acc: {validation_accuracy:.3f}"

    )


    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )


        torch.save({

            "model_state":
                model.state_dict(),

            "classes":
                CLASS_NAMES

        }, MODEL_PATH)


print("\nBest validation accuracy:",
      round(
          best_validation_accuracy,
          4
      ))


print(
    "Saved model:",
    MODEL_PATH
)


# ================================================================
# 24. TRAINING CURVES
# ================================================================

history_df = pd.DataFrame(
    history
)

plt.figure(
    figsize=(8,5)
)

plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"],
    marker="o",
    label="Train"
)

plt.plot(
    history_df["epoch"],
    history_df["validation_accuracy"],
    marker="o",
    label="Validation"
)

plt.title(
    "HERITAGE AI Training Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ================================================================
# 25. LOAD BEST MODEL
# ================================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()


# ================================================================
# 26. TEST EVALUATION
# ================================================================

y_true = []

y_pred = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE
        )

        outputs = model(
            images
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
            .cpu()
            .numpy()
        )


        y_pred.extend(
            predictions
        )

        y_true.extend(
            labels.numpy()
        )


test_accuracy = (
    accuracy_score(
        y_true,
        y_pred
    )
)


precision, recall, f1, _ = (
    precision_recall_fscore_support(

        y_true,
        y_pred,

        average="weighted",

        zero_division=0

    )
)


print("\n")
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    "Accuracy :",
    round(
        test_accuracy,
        4
    )
)

print(
    "Precision:",
    round(
        precision,
        4
    )
)

print(
    "Recall   :",
    round(
        recall,
        4
    )
)

print(
    "F1 Score :",
    round(
        f1,
        4
    )
)


print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ================================================================
# 27. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


plt.figure(
    figsize=(7,6)
)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    xticklabels=CLASS_NAMES,

    yticklabels=CLASS_NAMES

)


plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "HERITAGE AI Confusion Matrix"
)

plt.tight_layout()

plt.show()


# ================================================================
# 28. FEATURE EXTRACTION
# ================================================================

# EfficientNet feature extractor:
# We take features before the classification layer.

def extract_embedding(
    image_path
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    tensor = (
        evaluation_transforms(
            image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )


    with torch.no_grad():

        features = (
            model.features(
                tensor
            )
        )

        features = (
            model.avgpool(
                features
            )
        )

        features = torch.flatten(
            features,
            1
        )


    return (
        features
        .cpu()
        .numpy()[0]
    )


# ================================================================
# 29. BUILD HERITAGE EMBEDDING GALLERY
# ================================================================

gallery_paths = []

gallery_classes = []

gallery_embeddings = []


print("\n")
print("=" * 60)
print("BUILDING ARCHITECTURAL FEATURE GALLERY")
print("=" * 60)


for class_name in tqdm(
    CLASS_NAMES
):

    class_folder = (
        CLEAN_DIR /
        class_name
    )


    for image_path in sorted(
        class_folder.glob("*.jpg")
    ):

        try:

            embedding = extract_embedding(
                image_path
            )


            gallery_paths.append(
                str(image_path)
            )


            gallery_classes.append(
                class_name
            )


            gallery_embeddings.append(
                embedding
            )


        except Exception as e:

            print(
                "Embedding error:",
                image_path,
                e
            )


gallery_embeddings = normalize(
    np.vstack(
        gallery_embeddings
    )
)


print(
    "Gallery images:",
    len(gallery_paths)
)

print(
    "Embedding dimension:",
    gallery_embeddings.shape[1]
)


# ================================================================
# 30. NEAREST-NEIGHBOR RETRIEVAL
# ================================================================

nearest_neighbor = NearestNeighbors(

    n_neighbors=min(
        5,
        len(gallery_embeddings)
    ),

    metric="cosine"

)


nearest_neighbor.fit(
    gallery_embeddings
)


def retrieve_similar_images(
    image_path,
    k=5
):

    query_embedding = (
        extract_embedding(
            image_path
        )
    )


    query_embedding = normalize(

        query_embedding.reshape(
            1,
            -1
        )

    )


    distances, indices = (
        nearest_neighbor.kneighbors(

            query_embedding,

            n_neighbors=k

        )
    )


    results = []


    for distance, index in zip(

        distances[0],

        indices[0]

    ):

        results.append({

            "image":
                gallery_paths[index],

            "class":
                gallery_classes[index],

            "similarity":
                float(
                    1 - distance
                )

        })


    return pd.DataFrame(
        results
    )


# ================================================================
# 31. ARCHITECTURAL DNA
# ================================================================

# Important:
# We only assign descriptors that are justified by the predicted
# architectural category. Material, historical period, location etc.
# are NOT invented from pixels.

ARCHITECTURAL_DNA = {

    "Gopuram": {

        "scale":
            "large architectural structure",

        "geometry":
            "vertical and tiered",

        "structural_role":
            "temple gateway/tower",

        "common_visual_features":
            [
                "vertical mass",
                "stacked tiers",
                "decorative surfaces"
            ]

    },


    "Pillar": {

        "scale":
            "component-level",

        "geometry":
            "vertical elongated structure",

        "structural_role":
            "support element",

        "common_visual_features":
            [
                "base",
                "shaft",
                "capital",
                "carving"
            ]

    },


    "Mandapa": {

        "scale":
            "spatial architectural structure",

        "geometry":
            "hall-like",

        "structural_role":
            "pillared hall",

        "common_visual_features":
            [
                "columns",
                "roof",
                "open/closed hall space"
            ]

    },


    "Vimana": {

        "scale":
            "large architectural structure",

        "geometry":
            "vertical temple superstructure",

        "structural_role":
            "superstructure above sanctum",

        "common_visual_features":
            [
                "tiered form",
                "vertical symmetry",
                "ornamentation"
            ]

    },


    "Sculpture": {

        "scale":
            "detail-level architectural element",

        "geometry":
            "carved three-dimensional form",

        "structural_role":
            "decorative/figurative element",

        "common_visual_features":
            [
                "carving",
                "relief",
                "ornament"
            ]

    }

}


def create_architectural_dna(
    predicted_class,
    similarity_results
):

    dna = {

        "architectural_element":
            predicted_class,

        "visual_embedding":
            "EfficientNet-B0 feature vector",

        "structural_description":
            ARCHITECTURAL_DNA
            .get(
                predicted_class,
                {}
            ),

        "reference_count":
            len(
                similarity_results
            ),

        "best_reference_similarity":
            float(
                similarity_results[
                    "similarity"
                ].max()
            )
            if len(
                similarity_results
            ) > 0
            else 0.0,

        "material":
            "unknown",

        "historical_period":
            "unknown",

        "location":
            "unknown",

        "condition":
            "unknown"

    }


    return dna


# ================================================================
# 32. HERITAGE KNOWLEDGE GRAPH
# ================================================================

heritage_graph = nx.DiGraph()


for class_name in CLASS_NAMES:

    heritage_graph.add_node(

        class_name,

        node_type=
            "architectural_element"

    )


relationships = {

    "Gopuram": [

        "gateway",

        "vertical_structure",

        "temple_architecture"

    ],

    "Pillar": [

        "structural_support",

        "carved_element",

        "temple_architecture"

    ],

    "Mandapa": [

        "pillared_hall",

        "spatial_structure",

        "temple_architecture"

    ],

    "Vimana": [

        "sanctum_superstructure",

        "vertical_structure",

        "temple_architecture"

    ],

    "Sculpture": [

        "carved_element",

        "decorative_element",

        "temple_architecture"

    ]

}


for class_name, properties in relationships.items():

    if class_name not in CLASS_NAMES:

        continue


    for property_name in properties:

        heritage_graph.add_node(

            property_name,

            node_type=
                "heritage_property"

        )


        heritage_graph.add_edge(

            class_name,

            property_name,

            relation=
                "has_property"

        )


print("\n")
print("=" * 60)
print("HERITAGE KNOWLEDGE GRAPH")
print("=" * 60)

print(
    "Nodes:",
    heritage_graph.number_of_nodes()
)

print(
    "Edges:",
    heritage_graph.number_of_edges()
)


# ================================================================
# 33. PROPOSED EWAR FRAMEWORK
# ================================================================

def calculate_ewar(

    predicted_class,

    confidence,

    similarity_results,

    dna_match,

    knowledge_graph_match

):

    # ------------------------------------------------------------
    # Evidence components
    # ------------------------------------------------------------

    classifier_evidence = float(
        np.clip(
            confidence,
            0,
            1
        )
    )


    if len(
        similarity_results
    ) > 0:

        visual_evidence = float(

            similarity_results[
                "similarity"
            ]
            .mean()

        )

    else:

        visual_evidence = 0.0


    visual_evidence = float(

        np.clip(
            visual_evidence,
            0,
            1
        )

    )


    dna_evidence = float(

        np.clip(
            dna_match,
            0,
            1
        )

    )


    graph_evidence = float(

        np.clip(
            knowledge_graph_match,
            0,
            1
        )

    )


    # ------------------------------------------------------------
    # Prototype weighted evidence score
    # ------------------------------------------------------------
    #
    # These weights are RESEARCH HYPOTHESES.
    # They must be tuned and validated experimentally.
    #

    score = (

        0.35 * visual_evidence +

        0.30 * classifier_evidence +

        0.20 * dna_evidence +

        0.15 * graph_evidence

    )


    score = float(
        np.clip(
            score,
            0,
            1
        )
    )


    # ------------------------------------------------------------
    # Evidence status
    # ------------------------------------------------------------

    if score >= 0.80:

        status = (
            "Evidence-supported"
        )

    elif score >= 0.60:

        status = (
            "Evidence-assisted / uncertain"
        )

    else:

        status = (
            "Speculative - "
            "additional evidence required"
        )


    return {

        "visual_evidence":
            visual_evidence,

        "classifier_evidence":
            classifier_evidence,

        "architectural_dna_evidence":
            dna_evidence,

        "knowledge_graph_evidence":
            graph_evidence,

        "ewar_score":
            score,

        "status":
            status

    }


# ================================================================
# 34. COMPLETE PREDICTION FUNCTION
# ================================================================

def heritage_ai_predict(
    image_path
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    tensor = (
        evaluation_transforms(
            image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )


    # ------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------

    model.eval()


    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = (
            torch.softmax(
                outputs,
                dim=1
            )
        )


    predicted_index = int(

        probabilities
        .argmax(
            dim=1
        )
        .item()

    )


    predicted_class = (
        CLASS_NAMES[
            predicted_index
        ]
    )


    confidence = float(

        probabilities[
            0,
            predicted_index
        ]
        .item()

    )


    # ------------------------------------------------------------
    # Similar reference retrieval
    # ------------------------------------------------------------

    references = retrieve_similar_images(

        image_path,

        k=5

    )


    # ------------------------------------------------------------
    # Architectural DNA
    # ------------------------------------------------------------

    dna = create_architectural_dna(

        predicted_class,

        references

    )


    # ------------------------------------------------------------
    # DNA agreement
    # ------------------------------------------------------------

    if len(references) > 0:

        dna_match = float(

            (
                references["class"] ==
                predicted_class
            )
            .mean()

        )

    else:

        dna_match = 0.0


    # ------------------------------------------------------------
    # Knowledge graph evidence
    # ------------------------------------------------------------

    knowledge_graph_match = (

        1.0

        if predicted_class
        in heritage_graph.nodes

        else 0.0

    )


    # ------------------------------------------------------------
    # EWAR
    # ------------------------------------------------------------

    evidence = calculate_ewar(

        predicted_class,

        confidence,

        references,

        dna_match,

        knowledge_graph_match

    )


    # ------------------------------------------------------------
    # Evidence-based reconstruction proposal
    # ------------------------------------------------------------

    reconstruction = {

        "type":
            "reference-based proposal",

        "status":
            evidence[
                "status"
            ],

        "explanation":
            (
                "The system retrieves visually similar "
                "heritage references and uses them as "
                "supporting evidence. This output should "
                "not be presented as historically authentic "
                "without independent historical evidence."
            ),

        "reference_images":
            references[
                "image"
            ].tolist()

    }


    return {

        "prediction":
            predicted_class,

        "confidence":
            confidence,

        "architectural_dna":
            dna,

        "references":
            references,

        "evidence":
            evidence,

        "reconstruction":
            reconstruction

    }


# ================================================================
# 35. GRAD-CAM XAI
# ================================================================

# Hooks for final EfficientNet feature layer

gradcam_activations = None

gradcam_gradients = None


def activation_hook(
    module,
    input_data,
    output_data
):

    global gradcam_activations

    gradcam_activations = (
        output_data
    )


def gradient_hook(
    module,
    grad_input,
    grad_output
):

    global gradcam_gradients

    gradcam_gradients = (
        grad_output[0]
    )


target_layer = (
    model.features[-1]
)


hook_forward = (
    target_layer.register_forward_hook(
        activation_hook
    )
)


hook_backward = (
    target_layer.register_full_backward_hook(
        gradient_hook
    )
)


def generate_gradcam(
    image_path
):

    global gradcam_activations
    global gradcam_gradients


    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    tensor = (
        evaluation_transforms(
            image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )


    model.zero_grad()


    output = model(
        tensor
    )


    predicted_index = (
        output.argmax(
            dim=1
        )
    )


    output[
        0,
        predicted_index
    ].backward()


    activations = (
        gradcam_activations
        .detach()[0]
    )


    gradients = (
        gradcam_gradients
        .detach()[0]
    )


    weights_cam = (
        gradients
        .mean(
            dim=(1,2)
        )
    )


    cam = torch.relu(

        (
            weights_cam
            [:, None, None]
            *
            activations
        )
        .sum(
            dim=0
        )

    )


    cam = (
        cam.cpu()
        .numpy()
    )


    cam -= cam.min()


    if cam.max() > 0:

        cam /= cam.max()


    heatmap = Image.fromarray(

        np.uint8(
            cam * 255
        )

    ).resize(

        image.size,

        Image.Resampling.BILINEAR

    )


    return (
        image,
        heatmap,
        CLASS_NAMES[
            predicted_index.item()
        ]
    )


# ================================================================
# 36. RUN EXAMPLE PREDICTION
# ================================================================

example_images = list(

    CLEAN_DIR.rglob(
        "*.jpg"
    )

)


if len(
    example_images
) > 0:


    EXAMPLE_IMAGE = (
        str(
            example_images[0]
        )
    )


    print("\n")
    print("=" * 60)
    print("EXAMPLE HERITAGE AI PREDICTION")
    print("=" * 60)


    result = heritage_ai_predict(

        EXAMPLE_IMAGE

    )


    print(
        "\nInput:",
        EXAMPLE_IMAGE
    )


    print(
        "\nPredicted architectural element:",
        result["prediction"]
    )


    print(
        "Confidence:",
        f"{result['confidence']:.2%}"
    )


    print("\n")
    print(
        "Architectural DNA:"
    )


    print(
        json.dumps(
            result[
                "architectural_dna"
            ],
            indent=2
        )
    )


    print("\n")
    print(
        "EWAR Evidence Analysis:"
    )


    print(
        json.dumps(
            result[
                "evidence"
            ],
            indent=2
        )
    )


    print("\n")
    print(
        "Top reference images:"
    )


    print(
        result[
            "references"
        ].to_string(
            index=False
        )
    )


# ================================================================
# 37. GRAD-CAM VISUALIZATION
# ================================================================

if len(
    example_images
) > 0:


    original_image, heatmap, predicted_class = (
        generate_gradcam(
            EXAMPLE_IMAGE
        )
    )


    plt.figure(
        figsize=(12,5)
    )


    plt.subplot(
        1,
        2,
        1
    )


    plt.imshow(
        original_image
    )


    plt.title(
        f"Original\n{predicted_class}"
    )


    plt.axis(
        "off"
    )


    plt.subplot(
        1,
        2,
        2
    )


    plt.imshow(
        original_image
    )


    plt.imshow(
        heatmap,
        alpha=0.45,
        cmap="jet"
    )


    plt.title(
        "Grad-CAM XAI"
    )


    plt.axis(
        "off"
    )


    plt.tight_layout()

    plt.show()


# ================================================================
# 38. SAVE KNOWLEDGE GRAPH
# ================================================================

GRAPH_PATH = (
    OUTPUT_DIR /
    "heritage_knowledge_graph.gexf"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


nx.write_gexf(

    heritage_graph,

    GRAPH_PATH

)


# ================================================================
# 39. SAVE PROJECT SUMMARY
# ================================================================

project_summary = {

    "project":
        "HERITAGE AI",

    "framework":
        "HADAR + EWAR prototype",

    "classes":
        CLASS_NAMES,

    "total_clean_images":
        int(
            len(metadata)
        ),

    "model":
        "EfficientNet-B0 transfer learning",

    "image_size":
        IMAGE_SIZE,

    "embedding":
        "EfficientNet feature embedding",

    "retrieval":
        "Cosine similarity with nearest neighbours",

    "knowledge_graph":
        "NetworkX",

    "xai":
        "Grad-CAM",

    "proposed_reasoning":
        "Evidence Weighted Architectural Reasoning (EWAR)",

    "reconstruction":
        "Reference-based proposal",

    "license_status":
        "Google-derived images remain unverified",

    "historical_authenticity":
        "Not established by the model"

}


SUMMARY_PATH = (
    OUTPUT_DIR /
    "heritage_ai_summary.json"
)


with open(
    SUMMARY_PATH,
    "w"
) as file:

    json.dump(
        project_summary,
        file,
        indent=4
    )


# ================================================================
# 40. FINAL REPORT
# ================================================================

print("\n")
print("=" * 60)
print("HERITAGE AI PIPELINE COMPLETE")
print("=" * 60)

print(
    "\nClasses:",
    CLASS_NAMES
)

print(
    "Clean images:",
    len(metadata)
)

print(
    "Test Accuracy:",
    round(
        test_accuracy,
        4
    )
)

print(
    "Weighted F1:",
    round(
        f1,
        4
    )
)

print(
    "\nImportant files:"
)

print(
    "1. Metadata:",
    METADATA_PATH
)

print(
    "2. Split metadata:",
    BASE_DIR /
    "heritage_split_metadata.csv"
)

print(
    "3. Trained model:",
    MODEL_PATH
)

print(
    "4. Contact sheets:",
    CONTACT_DIR
)

print(
    "5. Knowledge graph:",
    GRAPH_PATH
)

print(
    "6. Project summary:",
    SUMMARY_PATH
)


print("\n")
print("=" * 60)
print("RESEARCH PIPELINE")
print("=" * 60)

print(
"""
Image
  ↓
Cleaning
  ↓
EfficientNet Transfer Learning
  ↓
Architectural Classification
  ↓
Architectural Feature Embedding
  ↓
Architectural DNA
  ↓
Heritage Knowledge Graph
  ↓
Evidence Retrieval
  ↓
EWAR Evidence Score
  ↓
Reference-based Reconstruction Proposal
  ↓
Grad-CAM Explainable AI
"""
)

print(
"\nNEXT RESEARCH STEP:"
)

print(
"""
1. Manually inspect the contact sheets.
2. Remove wrongly labelled / unusable photographs.
3. Verify original source and license for each image.
4. Add the Vimana class.
5. Add more verified heritage references.
6. Compare EfficientNet with ViT/other baselines.
7. Experimentally validate EWAR instead of assuming its weights.
8. Develop the reconstruction module after sufficient reference data exists.
"""
)

print("\nDONE.")
