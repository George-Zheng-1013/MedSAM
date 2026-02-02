import os
import glob
import random
import monai
from os import makedirs
from os.path import join
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from segment_anything import sam_model_registry
import cv2
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    help="Path to the data root directory.",
    required=True,
)
parser.add_argument(
    "-medsam_checkpoint", type=str, help="Path to the MedSAM checkpoint.", required=True
)
parser.add_argument(
    "-work_dir",
    type=str,
    default="finetune_point_prompt",
    help="Path to where the checkpoints and logs are saved.",
)
parser.add_argument(
    "-max_epochs", type=int, default=1000, help="Maximum number of epochs."
)
parser.add_argument("-batch_size", type=int, default=16, help="Batch size.")
parser.add_argument(
    "-num_workers", type=int, default=8, help="Number of data loader workers."
)
parser.add_argument(
    "-resume", type=str, default=None, help="Path to the checkpoint to resume from."
)
parser.add_argument(
    "-lr", type=float, default=0.00005, help="learning rate (absolute lr)"
)
parser.add_argument("-weight_decay", type=float, default=0.01, help="Weight decay.")
parser.add_argument(
    "-seed", type=int, default=2023, help="Random seed for reproducibility."
)
parser.add_argument(
    "--disable_aug", action="store_true", help="Disable data augmentation."
)
parser.add_argument(
    "-val_list",
    type=str,
    default=None,
    help="Path to validation patient list (one patient ID per line).",
)
parser.add_argument(
    "-test_list",
    type=str,
    default=None,
    help="Path to test patient list (one patient ID per line).",
)
args = parser.parse_args()


# Dataset class
class NpyDataset(Dataset):
    def __init__(
        self,
        data_root,
        image_size=1024,
        data_aug=True,
        patient_list=None,
        exclude_patient_list=None,
        use_pts=False,
    ):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.pt_path = join(data_root, "pts")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        if patient_list:
            patient_set = set(patient_list)
            self.gt_path_files = [
                file
                for file in self.gt_path_files
                if os.path.basename(file).split("_")[0] in patient_set
            ]
        if exclude_patient_list:
            exclude_set = set(exclude_patient_list)
            self.gt_path_files = [
                file
                for file in self.gt_path_files
                if os.path.basename(file).split("_")[0] not in exclude_set
            ]
        self.image_size = image_size
        self.data_aug = data_aug
        self.use_pts = use_pts and os.path.isdir(self.pt_path)

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (H, W) or (H, W, 3)

        # Handle grayscale images
        if len(img_1024.shape) == 2:
            # Normalize to [0, 1]
            img_1024 = (img_1024 - img_1024.min()) / (
                img_1024.max() - img_1024.min() + 1e-8
            )
            # Resize to 1024x1024
            img_1024 = cv2.resize(
                img_1024, (1024, 1024), interpolation=cv2.INTER_LINEAR
            )
            # Convert grayscale to RGB by repeating the channel
            img_1024 = np.stack(
                [img_1024, img_1024, img_1024], axis=0
            )  # (3, 1024, 1024)
        else:
            # Normalize to [0, 1] if needed
            if np.max(img_1024) > 1.0:
                img_1024 = (img_1024 - img_1024.min()) / (
                    img_1024.max() - img_1024.min() + 1e-8
                )
            # convert the shape to (3, H, W)
            img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, H, W)
            # Resize to 1024x1024
            img_1024 = np.stack(
                [
                    cv2.resize(
                        img_1024[i], (1024, 1024), interpolation=cv2.INTER_LINEAR
                    )
                    for i in range(3)
                ],
                axis=0,
            )  # (3, 1024, 1024)

        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...]
        label_ids = np.unique(gt)[1:]
        try:
            gt2D = np.uint8(
                gt == random.choice(label_ids.tolist())
            )  # only one label, (256, 256)
        except:
            print(img_name, "label_ids.tolist()", label_ids.tolist())
            gt2D = np.uint8(gt == np.max(gt))  # only one label, (256, 256)
        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
        gt2D = np.uint8(gt2D > 0)

        # Resize ground truth to 1024x1024 to match image size
        gt2D = cv2.resize(gt2D, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        coords = None
        if self.use_pts:
            pts_file = join(self.pt_path, img_name.replace(".npy", ".npz"))
            if os.path.isfile(pts_file):
                pts_data = np.load(pts_file, "r", allow_pickle=True)
                point_coords = pts_data.get("point_coords", None)
                if point_coords is not None and len(point_coords) > 0:
                    coords = point_coords[0]

        if coords is None:
            y_indices, x_indices = np.where(gt2D > 0)
            if len(x_indices) > 0:
                x_point = np.random.choice(x_indices)
                y_point = np.random.choice(y_indices)
                coords = np.array([x_point, y_point])
            else:
                # If no foreground pixels, use center point
                coords = np.array([512, 512])

        ## Resize to 256x256 for final output
        gt2D_256 = cv2.resize(gt2D, (256, 256), interpolation=cv2.INTER_NEAREST)
        return {
            "image": torch.tensor(img_1024).float(),
            "gt2D": torch.tensor(gt2D_256[None, :, :]).long(),
            "coords": torch.tensor(coords[None, ...]).float(),
            "image_name": img_name,
        }


def load_patient_list(list_path):
    if not list_path:
        return None
    with open(list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def evaluate(medsam_model, data_loader):
    medsam_model.eval()
    dice_sum, iou_sum, total = 0.0, 0.0, 0
    eps = 1e-6
    for batch in data_loader:
        image = batch["image"].to(device)
        gt2D = batch["gt2D"].to(device).float()
        coords_torch = batch["coords"].to(device)
        labels_torch = torch.ones(coords_torch.shape[0], 1, device=device).long()
        point_prompt = (coords_torch, labels_torch)
        logits = medsam_model(image, point_prompt)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        intersection = (preds * gt2D).sum(dim=(1, 2, 3))
        pred_sum = preds.sum(dim=(1, 2, 3))
        gt_sum = gt2D.sum(dim=(1, 2, 3))
        dice = (2 * intersection + eps) / (pred_sum + gt_sum + eps)
        union = pred_sum + gt_sum - intersection
        iou = (intersection + eps) / (union + eps)
        dice_sum += dice.sum().item()
        iou_sum += iou.sum().item()
        total += preds.shape[0]
    medsam_model.train()
    return dice_sum / max(total, 1), iou_sum / max(total, 1)


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        freeze_image_encoder=False,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        self.freeze_image_encoder = freeze_image_encoder
        if self.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def forward(self, image, point_prompt):

        # do not compute gradients for pretrained img encoder and prompt encoder
        with torch.no_grad():
            image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
            # not need to convert box to 1024x1024 grid
            # bbox is already in 1024x1024
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=point_prompt,
                boxes=None,
                masks=None,
            )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )  # (B, 1, 256, 256)

        return low_res_masks


if __name__ == "__main__":
    data_root = args.tr_npy_path
    work_dir = args.work_dir
    num_epochs = args.max_epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    medsam_checkpoint = args.medsam_checkpoint
    data_aug = not args.disable_aug
    seed = args.seed
    device = "cuda:0"
    makedirs(work_dir, exist_ok=True)

    torch.cuda.empty_cache()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    sam_model = sam_model_registry["vit_b"](checkpoint=medsam_checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        freeze_image_encoder=True,
    )
    medsam_model = medsam_model.to(device)
    medsam_model.train()
    print(f"MedSAM size: {sum(p.numel() for p in medsam_model.parameters())}")

    optimizer = optim.AdamW(
        medsam_model.mask_decoder.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    val_list = load_patient_list(args.val_list)
    test_list = load_patient_list(args.test_list)
    exclude_list = set(val_list or []) | set(test_list or [])
    train_dataset = NpyDataset(
        data_root=data_root,
        data_aug=data_aug,
        exclude_patient_list=exclude_list if exclude_list else None,
        use_pts=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = None
    test_loader = None
    if val_list:
        val_dataset = NpyDataset(
            data_root=data_root, data_aug=False, patient_list=val_list, use_pts=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    if test_list:
        test_dataset = NpyDataset(
            data_root=data_root, data_aug=False, patient_list=test_list, use_pts=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    resume = args.resume
    if resume:
        checkpoint = torch.load(resume)
        medsam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        print(f"Loaded checkpoint from epoch {start_epoch}, best loss: {best_loss:.4f}")
    else:
        start_epoch = 0
        best_loss = 1e10
    torch.cuda.empty_cache()

    epoch_time = []
    losses = []
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = [1e10 for _ in range(len(train_loader))]
        epoch_start_time = time()
        pbar = tqdm(train_loader)
        for step, batch in enumerate(pbar):
            image = batch["image"]
            gt2D = batch["gt2D"]
            coords_torch = batch["coords"]  # (B, 2)
            optimizer.zero_grad()
            labels_torch = torch.ones(coords_torch.shape[0]).long()  # (B,)
            labels_torch = labels_torch.unsqueeze(1)  # (B, 1)
            image, gt2D = image.to(device), gt2D.to(device).float()
            coords_torch, labels_torch = coords_torch.to(device), labels_torch.to(
                device
            )
            point_prompt = (coords_torch, labels_torch)
            medsam_lite_pred = medsam_model(image, point_prompt)
            loss = seg_loss(medsam_lite_pred, gt2D) + ce_loss(medsam_lite_pred, gt2D)
            epoch_loss[step] = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(
                f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}"
            )

        epoch_end_time = time()
        epoch_time.append(epoch_end_time - epoch_start_time)
        epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
        losses.append(epoch_loss_reduced)
        model_weights = medsam_model.state_dict()
        checkpoint = {
            "model": model_weights,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "loss": epoch_loss_reduced,
            "best_loss": best_loss,
        }
        if epoch_loss_reduced < best_loss:
            print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
            best_loss = epoch_loss_reduced
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, join(work_dir, "medsam_point_prompt_best.pth"))

        torch.save(checkpoint, join(work_dir, "medsam_point_prompt_latest.pth"))

        if val_loader is not None:
            val_dice, val_iou = evaluate(medsam_model, val_loader)
            print(f"Validation Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.plot(losses)
        ax1.set_title("Dice + Cross Entropy Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.plot(epoch_time)
        ax2.set_title("Epoch Running Time")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Time (s)")
        fig.savefig(join(work_dir, "medsam_point_prompt_loss_time.png"))

        epoch_loss_reduced = 1e10

    if test_loader is not None:
        test_dice, test_iou = evaluate(medsam_model, test_loader)
        print(f"Test Dice: {test_dice:.4f}, IoU: {test_iou:.4f}")
