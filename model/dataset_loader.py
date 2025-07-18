import os
import random
from typing import List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset


class FewShotDataset(Dataset):

    IMG_EXT = (".jpg", ".jpeg", ".png")

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Sorted class list so ordering is stable
        self.class_folders: List[str] = sorted(
            f for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
        )

        # Index all image filenames per class once at init
        self.class_to_imgs: Dict[str, List[str]] = {}
        for cls in self.class_folders:
            cls_dir = os.path.join(root_dir, cls)
            imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(self.IMG_EXT)]
            if len(imgs) == 0:
                raise RuntimeError(f"Class folder '{cls}' contains no images.")
            self.class_to_imgs[cls] = imgs

    # Standard Dataset interface (rarely used here)
    def __len__(self):
        return sum(len(v) for v in self.class_to_imgs.values())

    def __getitem__(self, idx):
        raise NotImplementedError("Episodic dataset – call get_episode() instead.")

    # Episodic sampler (now allows replacement)
    def get_episode(self, n_way: int, k_shot: int, q_query: int):
        """Return tensors `(sup_x, sup_y, qry_x, qry_y)` for **one** episode.

        • Chooses *n_way* classes at random (requires at least 1 image/class).
        • If a class has < `k_shot + q_query` images we sample **with
          replacement** so list is never empty.
        """
        if len(self.class_folders) < n_way:
            raise RuntimeError(
                f"Dataset has only {len(self.class_folders)} classes but n_way={n_way} requested.")

        # Pick n_way classes randomly (no other restriction now)
        selected_classes = random.sample(self.class_folders, n_way)

        sup_imgs, sup_labels = [], []
        qry_imgs, qry_labels = [], []

        for label_idx, cls in enumerate(selected_classes):
            cls_dir = os.path.join(self.root_dir, cls)
            imgs = self.class_to_imgs[cls]

            # If folder too small, use replacement to get enough samples
            if len(imgs) >= k_shot + q_query:
                imgs_shuffled = random.sample(imgs, len(imgs))
                sup_files = imgs_shuffled[:k_shot]
                qry_files = imgs_shuffled[k_shot:k_shot + q_query]
            else:
                sup_files = random.choices(imgs, k=k_shot)
                qry_files = random.choices(imgs, k=q_query)

            # Load images & apply transforms
            for f in sup_files:
                img = Image.open(os.path.join(cls_dir, f)).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                sup_imgs.append(img)
                sup_labels.append(label_idx)

            for f in qry_files:
                img = Image.open(os.path.join(cls_dir, f)).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                qry_imgs.append(img)
                qry_labels.append(label_idx)

        # Stack into tensors (never empty now)
        sup_x = torch.stack(sup_imgs)
        sup_y = torch.tensor(sup_labels)
        qry_x = torch.stack(qry_imgs)
        qry_y = torch.tensor(qry_labels)

        return sup_x, sup_y, qry_x, qry_y