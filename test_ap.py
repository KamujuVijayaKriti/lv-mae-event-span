import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

import models_mae
from data.CombinedData import CombinedData


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = (
    "./logs/"
    "prod_combined_b2_e150_modelmae_vit_base_custom_in1024_mask0.4_"
    "wd0.05_warmup40_d32_heads8_dec_embed_dim512_dec_d8_dec_heads16_"
    "distributed_False_cosloss_False_ms_True/checkpoint-149.pth"
)

TRAIN_CSV = (
    "data/"
    "xd_train.csv"
)

VAL_CSV = (
    "data/"
    "xd_test.csv"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2
EPOCHS = 60
LR = 1e-4
WEIGHT_DECAY = 1e-3
NUM_CLASSES = 7   # XD-Violence; override with LVMAE_NUM_CLASSES
INPUT_DIM = 1024
NUM_HEADS = 8

# IMPORTANT:
# For AP evaluation, use the full/unmasked embeddings.
MASK_RATIO = 0.0
import os; CHECKPOINT = os.environ.get("LVMAE_CKPT", CHECKPOINT); torch.manual_seed(int(os.environ.get("LVMAE_SEED", "0"))); print("AP probe on:", CHECKPOINT, flush=True)
TRAIN_CSV = os.environ.get("LVMAE_TRAIN_CSV", TRAIN_CSV); VAL_CSV = os.environ.get("LVMAE_VAL_CSV", VAL_CSV); NUM_CLASSES = int(os.environ.get("LVMAE_NUM_CLASSES", NUM_CLASSES)); print("probe data:", TRAIN_CSV, VAL_CSV, "classes:", NUM_CLASSES, flush=True)

# Keep sequence length consistent with the trained model.
NUM_PATCHES = 256

# Do not randomly shorten videos during AP evaluation.
# The dataset will still use the existing maximum length.
DURATION_MAX = 1280


# ============================================================
# XD-Violence class labels (same order as data_preprocess/build_xd_csv.py)
# ============================================================

LABEL_MAP = {
    "Normal": 0,
    "Fighting": 1,
    "Shooting": 2,
    "Riot": 3,
    "Abuse": 4,
    "Car accident": 5,
    "Explosion": 6,
}


# ============================================================
# Load pretrained LV-MAE
# ============================================================

def load_pretrained_model():

    print("\nLoading checkpoint:")
    print(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu"
    )

    args = checkpoint["args"]

    print("\nCheckpoint information:")
    print("Epoch:", checkpoint["epoch"])
    print("Model:", args.model)
    print("Input dimension:", args.input_size)
    print("Depth:", args.depth)
    print("Heads:", args.num_heads)
    print("Num patches:", args.num_patches)
    print("Mask ratio:", args.mask_ratio)
    print(
        "Masking strategy:",
        getattr(args, "masking_strategy", "NOT FOUND")
    )

    model = models_mae.__dict__[args.model](
        args=args,
        norm_pix_loss=args.norm_pix_loss
    )

    model.load_state_dict(
        checkpoint["model"],
        strict=True
    )

    model.to(DEVICE)

    # Freeze entire LV-MAE encoder.
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    print("\nLV-MAE checkpoint loaded successfully.")
    print("Device:", DEVICE)

    return model


# ============================================================
# Attentive Probing
# ============================================================

class AttentiveProbing(nn.Module):

    def __init__(
        self,
        encoder_model,
        input_dim=1024,
        num_classes=10,
        num_heads=8
    ):
        super().__init__()

        # ------------------------------------------------------------
        # Frozen pretrained LV-MAE encoder
        # ------------------------------------------------------------
        self.encoder_model = encoder_model

        for param in self.encoder_model.parameters():
            param.requires_grad = False

        # ------------------------------------------------------------
        # One trainable Transformer block
        # Same architecture as reference test_ap_movie_clip.py
        # ------------------------------------------------------------
        self.blocks = nn.ModuleList([
            models_mae.Block(
                input_dim,
                num_heads,
                4.0,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
            )
        ])

        # ------------------------------------------------------------
        # Trainable normalization
        # ------------------------------------------------------------
        self.norm = nn.LayerNorm(input_dim)

        # ------------------------------------------------------------
        # Linear classifier
        # ------------------------------------------------------------
        self.linear = nn.Linear(
            input_dim,
            num_classes
        )

    def forward(
        self,
        x,
        attn_mask,
        ids_shuffle,
        length
    ):

        # ------------------------------------------------------------
        # Frozen pretrained LV-MAE encoder
        # ------------------------------------------------------------
        with torch.no_grad():

            x = self.encoder_model.forward_encoder(
                x,
                attn_mask,
                ids_shuffle=None,
                length=length
            )

        # ------------------------------------------------------------
        # One trainable attentive probing Transformer block
        # ------------------------------------------------------------
        for blk in self.blocks:

            x = blk(
                x,
                attn_mask
            )

        # ------------------------------------------------------------
        # Normalize
        # ------------------------------------------------------------
        x = self.norm(x)

        # ------------------------------------------------------------
        # CLS token
        # ------------------------------------------------------------
        cls_token = x[:, 0]

        # ------------------------------------------------------------
        # Linear classifier
        # ------------------------------------------------------------
        return self.linear(cls_token)# ============================================================
# Dataset
# ============================================================

def create_dataloaders():

    print("\nLoading datasets...")

    train_dataset = CombinedData(
        csv_file=TRAIN_CSV,
        mask_ratio=MASK_RATIO,
        num_patches=NUM_PATCHES,
        duration_max=DURATION_MAX,
        importance_masking_strategy=False
    )

    val_dataset = CombinedData(
        csv_file=VAL_CSV,
        duration_min=DURATION_MAX,
        mask_ratio=MASK_RATIO,
        num_patches=NUM_PATCHES,
        duration_max=DURATION_MAX,
        importance_masking_strategy=False
    )

    print("Training videos:", len(train_dataset))
    print("Validation videos:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    probe,
    dataloader,
    optimizer,
    criterion
):

    probe.train()

    # Keep pretrained encoder in evaluation mode.
    probe.encoder_model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):

        inputs = batch[
            "normalized_stacked_embeddings"
        ].to(
            DEVICE,
            non_blocking=True
        )

        attn_mask = batch[
            "attn_mask"
        ].to(
            DEVICE,
            non_blocking=True
        )

        ids_shuffle = batch[
            "ids_shuffle"
        ].to(
            DEVICE,
            non_blocking=True
        )

        length = batch[
            "length"
        ].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch[
            "label"
        ].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = probe(
            inputs,
            attn_mask,
            ids_shuffle,
            length
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if batch_idx % 20 == 0:

            print(
                f"  Batch {batch_idx}/{len(dataloader)} "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(dataloader)

    accuracy = correct / total

    return avg_loss, accuracy


# ============================================================
# Validation
# ============================================================

def evaluate(
    probe,
    dataloader,
    criterion
):

    probe.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():

        for batch in dataloader:

            inputs = batch[
                "normalized_stacked_embeddings"
            ].to(
                DEVICE,
                non_blocking=True
            )

            attn_mask = batch[
                "attn_mask"
            ].to(
                DEVICE,
                non_blocking=True
            )

            ids_shuffle = batch[
                "ids_shuffle"
            ].to(
                DEVICE,
                non_blocking=True
            )

            length = batch[
                "length"
            ].to(
                DEVICE,
                non_blocking=True
            )

            labels = batch[
                "label"
            ].to(
                DEVICE,
                non_blocking=True
            )

            logits = probe(
                inputs,
                attn_mask,
                ids_shuffle,
                length
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            for label, prediction in zip(
                labels,
                predictions
            ):

                label = label.item()
                prediction = prediction.item()

                class_total[label] += 1

                if label == prediction:
                    class_correct[label] += 1

    avg_loss = total_loss / len(dataloader)

    accuracy = correct / total

    return (
        avg_loss,
        accuracy,
        class_correct,
        class_total
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("LV-MAE Attentive Probing")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # Load pretrained LV-MAE
    # --------------------------------------------------------

    model = load_pretrained_model()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, val_loader = create_dataloaders()

    # --------------------------------------------------------
    # Attentive probing
    # --------------------------------------------------------

    probe = AttentiveProbing(
        encoder_model=model,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        num_heads=NUM_HEADS
    ).to(DEVICE)

    # --------------------------------------------------------
    # Count parameters
    # --------------------------------------------------------

    trainable_params = sum(
        p.numel()
        for p in probe.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in probe.parameters()
    )

    print("\nAP model:")
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            probe.parameters()
        ),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # --------------------------------------------------------
    # Best model tracking
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    os.makedirs(
        "./logs/ap_60ep",
        exist_ok=True
    )

    best_path = (
        "./logs/ap_60ep/" +
        os.environ.get("LVMAE_AP_NAME", "best_attentive_probe_60ep.pth")
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        print("\n" + "=" * 70)
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )
        print("=" * 70)

        train_loss, train_accuracy = train_one_epoch(
            probe,
            train_loader,
            optimizer,
            criterion
        )

        print(
            f"\nTrain Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_accuracy,
            class_correct,
            class_total
        ) = evaluate(
            probe,
            val_loader,
            criterion
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # Save best
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": probe.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_accuracy": val_accuracy
                },
                best_path
            )

            print(
                "\n*** Best AP model saved ***"
            )

            print(
                "Path:",
                best_path
            )

        scheduler.step()

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best test accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Best test accuracy (%): "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        "Best model:",
        best_path
    )


if __name__ == "__main__":
    main()
