import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from models.phys_mamba_fusion import PhysMambaFusion
from losses.physics_loss import PhysMambaFusionLoss


class DummyIndustrialDataset(Dataset):
    """
    Placeholder dataset. Replace with real cylinder block + DIC data.
    img:     (3, 224, 224)
    dic_seq: (T, 2, 56, 56)  displacement fields
    targets: bbox, cls, risk, k_factor
    """
    def __init__(self, size=64, img_size=224, dic_size=56, T=8):
        self.size = size
        self.img_size = img_size
        self.dic_size = dic_size
        self.T = T

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.randn(3, self.img_size, self.img_size)
        dic_seq = torch.randn(self.T, 2, self.dic_size, self.dic_size)
        targets = {
            'bbox': torch.tensor([0.5, 0.5, 0.1, 0.1]),
            'cls': torch.tensor([1.0]),
            'risk': torch.tensor([0.7]),
            'k_factor': torch.tensor([1.2]),
        }
        return img, dic_seq, targets


def collate_fn(batch):
    imgs, dic_seqs, targets_list = zip(*batch)
    imgs = torch.stack(imgs)
    dic_seqs = torch.stack(dic_seqs)
    targets = {
        k: torch.stack([t[k] for t in targets_list])
        for k in targets_list[0]
    }
    return imgs, dic_seqs, targets


def train(
    epochs=10,
    batch_size=4,
    lr=1e-4,
    d_model=128,
    device='cpu',
    save_path='checkpoints/phys_mamba_fusion.pth',
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = DummyIndustrialDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = PhysMambaFusion(d_model=d_model).to(device)
    criterion = PhysMambaFusionLoss(lambda_risk=1.0, lambda_phys=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, dic_seqs, targets in loader:
            imgs = imgs.to(device)
            dic_seqs = dic_seqs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()
            preds = model(imgs, dic_seqs)
            losses = criterion(preds, targets)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += losses['total'].item()

        scheduler.step()
        avg = total_loss / len(loader)
        print(f'Epoch [{epoch}/{epochs}]  loss={avg:.4f}')

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')


if __name__ == '__main__':
    train()
