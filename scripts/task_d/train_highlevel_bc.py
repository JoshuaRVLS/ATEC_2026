from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from highlevel_utils import COMMAND_DIM, FEATURE_DIM


class HighLevelPolicy(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, output_dim: int = COMMAND_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        self.command_scale = nn.Parameter(torch.tensor([1.0, 1.0, 0.6]), requires_grad=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features) * self.command_scale.to(device=features.device, dtype=features.dtype)


def load_dataset(path: str, min_score: float | None = None):
    features = []
    commands = []

    with h5py.File(path, "r") as h5:
        for key in sorted(h5.keys()):
            group = h5[key]
            final_score = float(group.attrs.get("final_score", 0.0))
            if min_score is not None and final_score < min_score:
                continue
            features.append(group["features"][:])
            commands.append(group["commands"][:])

    if not features:
        raise ValueError(f"No episodes found in {path}. Try lowering --min_score.")

    x = np.concatenate(features, axis=0).astype(np.float32)
    y = np.concatenate(commands, axis=0).astype(np.float32)

    if x.shape[1] != FEATURE_DIM:
        raise ValueError(f"Feature dim mismatch: got {x.shape[1]}, expected {FEATURE_DIM}")
    if y.shape[1] != COMMAND_DIM:
        raise ValueError(f"Command dim mismatch: got {y.shape[1]}, expected {COMMAND_DIM}")

    return x, y


def main():
    parser = argparse.ArgumentParser(description="Train Task D high-level BC policy.")
    parser.add_argument("--data", type=str, default="datasets/task_d_highlevel/demos.hdf5")
    parser.add_argument("--out", type=str, default="demo/high_level_policy.pt")
    parser.add_argument("--ckpt", type=str, default="logs/task_d_highlevel/high_level_policy_ckpt.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--min_score", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    x_np, y_np = load_dataset(args.data, min_score=args.min_score)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0).clamp_min(1e-4)
    x = (x - x_mean) / x_std

    dataset = TensorDataset(x, y)
    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = HighLevelPolicy(hidden_dim=args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.shape[0]
            train_count += batch_x.shape[0]

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.inference_mode():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(args.device)
                batch_y = batch_y.to(args.device)
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                val_loss += loss.item() * batch_x.shape[0]
                val_count += batch_x.shape[0]

        train_loss /= max(train_count, 1)
        val_loss /= max(val_count, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"[train] epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "feature_mean": x_mean,
            "feature_std": x_std,
            "feature_dim": FEATURE_DIM,
            "command_dim": COMMAND_DIM,
            "hidden_dim": args.hidden_dim,
            "best_val_loss": best_val,
        },
        ckpt_path,
    )

    class NormalizedPolicy(nn.Module):
        def __init__(self, policy: nn.Module, mean: torch.Tensor, std: torch.Tensor):
            super().__init__()
            self.policy = policy
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.policy((features - self.mean) / self.std)

    export_model = NormalizedPolicy(model.cpu().eval(), x_mean.cpu(), x_std.cpu()).eval()
    example = torch.zeros((1, FEATURE_DIM), dtype=torch.float32)
    scripted = torch.jit.trace(export_model, example)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))

    print(f"[train] saved checkpoint: {ckpt_path}")
    print(f"[train] exported TorchScript policy: {out_path}")
    print(f"[train] best_val_loss={best_val:.6f}, samples={len(dataset)}")


if __name__ == "__main__":
    main()
