import argparse
import math
import os
import multiprocessing
import time
import random

import numpy as np
import torch
import torchvision
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.distributed import init_process_group, destroy_process_group, barrier, all_reduce, ReduceOp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from gram_newton_schulz import Muon
import wandb

from model import create_model, DiT, BASE_WIDTH

def unwrap_compiled(module: torch.nn.Module) -> torch.nn.Module:
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    raw_model: DiT,
    ema_model: AveragedModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    timed_training_seconds: float,
) -> None:
    ema = unwrap_compiled(ema_model)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model": raw_model.state_dict(),
            "ema_model": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "timed_training_seconds": timed_training_seconds,
        },
        path,
    )


def load_checkpoint(
    path: str,
    raw_model: DiT,
    ema_model: AveragedModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, float]:
    # Torch >=2.6 defaults to weights_only=True, which cannot deserialize
    # optimizer state for checkpoints that include pickled Python globals.
    # Resuming training needs full state, so we opt into full loading for
    # user-provided local checkpoints.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw_model.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return (
        int(checkpoint["step"]),
        int(checkpoint.get("epoch", 0)),
        float(checkpoint.get("timed_training_seconds", 0.0)),
    )


def generate_x_pos(image_size: tuple[int, int], patch_size: int, device: torch.device) -> torch.Tensor:
    grid_h = math.ceil(image_size[0] / patch_size)
    grid_w = math.ceil(image_size[1] / patch_size)
    xlim, ylim = math.sqrt(grid_w / grid_h), math.sqrt(grid_h / grid_w)
    rows = torch.linspace(-ylim, ylim, grid_h, dtype=torch.float32, device=device)
    cols = torch.linspace(-xlim, xlim, grid_w, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1).reshape(1, grid_h * grid_w, 2)


@torch.no_grad()
def sample_images(model, device, n, steps=50, seed=None, classes=None):
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None

    if classes is None:
        classes = torch.randint(0, 1000, (n,), device=device, generator=g)

    raw_model = model.module if hasattr(model, 'module') else model
    x_pos = generate_x_pos(raw_model.config.image_size, raw_model.config.patch_size, device)

    model.eval()
    img_size = (3, raw_model.config.image_size[0], raw_model.config.image_size[1])
    x = torch.randn((n, *img_size), device=device, generator=g)
    step_size = 1.0 / steps
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for step in range(steps):
            t = torch.full((n,), float(step) / steps, device=device)
            x_pred = model(x, x_pos, t, classes)
            v = (x_pred - x) / (1 - t.view(-1, 1, 1, 1))
            x = x + v * step_size

    x = ((x + 1) / 2).clamp(min=0, max=1)
    return x


@torch.no_grad()
def compute_fid(model, device, val_loader, master_process=True, rank=0):
    """
    Computes FID. In DDP mode, this MUST be called by all ranks.
    Each rank will process a subset of the data, and torchmetrics will 
    automatically sync the features across all GPUs during .compute().
    """
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for i, batch in enumerate(tqdm(val_loader, "Computing FID", disable=not master_process)):
        images = batch["image"]
        batch_size = images.shape[0]
        fid.update(images_uint8_to_device_float(images, device, normalize=False), real=True)

        classes = batch["label"].to(device, non_blocking=True)
        # Different seed per rank to ensure we generate different images across GPUs
        samples = sample_images(
            model, device, n=batch_size, seed=i * 1000 + rank, classes=classes
        )
        fid.update(samples, real=False)

    # .compute() performs an all-gather across all DDP ranks
    return fid.compute().item()


def sample_timestep_logit_normal(num: int, device: torch.device, mean: float, std: float) -> torch.Tensor:
    logit_t = torch.randn(num, device=device) * std + mean
    return torch.sigmoid(logit_t)


def images_uint8_to_device_float(images: torch.Tensor, device: torch.device, normalize: bool) -> torch.Tensor:
    # move uint8 image to GPU first to reduce bandwidth
    # then convert to float32 and rescale to [-1, 1] if needed on GPU
    images = images.to(device, non_blocking=True).to(torch.float32)
    if normalize:
        return images.mul_(1.0 / 127.5).sub_(1.0)
    return images.mul_(1.0 / 255.0)


class HFImageTransform:
    """Batch image preprocessing for Hugging Face Dataset.set_transform."""

    def __init__(self, random_flip=False):
        self.random_flip = random_flip

    def __call__(self, batch):
        images = batch["image"]
        is_batched = isinstance(images, list)
        if not is_batched:
            images = [images]

        arrays = []
        for image in images:
            if image.mode != "RGB":
                image = image.convert("RGB")
            arrays.append(np.asarray(image))

        tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
        if self.random_flip:
            flip_mask = torch.rand(tensor.shape[0]) < 0.5
            tensor[flip_mask] = tensor[flip_mask].flip(-1)

        batch["image"] = tensor if is_batched else tensor[0]
        return batch


def hidden_lr_scale(name: str, p: torch.Tensor):
    # muP scaling of the learning rate based on layer width
    unscaled_params = (
        "class_embedder.weight",
        "timestep_embedder.input_proj.weight",
        "final_layer.linear.weight",
    )
    if p.ndim != 2 or name in unscaled_params or name.startswith("patch_embed."):
        return 1.0
    return BASE_WIDTH / p.shape[1]


def qkv_split_fn(param: torch.Tensor):
    """
    Split Wqkv into [Wq, Wk, Wv].

    Assumes param has shape (3*hidden_dim, hidden_dim) where the first dimension
    is concatenated [Q, K, V] weights.
    """
    hidden_dim = param.size(1)
    Wq = param[:hidden_dim, :]
    Wk = param[hidden_dim:2*hidden_dim, :]
    Wv = param[2*hidden_dim:, :]
    return [Wq, Wk, Wv]


def qkv_recombine_fn(splits):
    """Recombine [Wq, Wk, Wv] back into Wqkv."""
    return torch.cat(splits, dim=0)


def setup_optimizer(raw_model: DiT, use_muon: bool, lr: float, muon_lr: float, ddp: bool):
    qkv_params = []
    muon_params = []
    adamw_params_by_lr_scale = {}
    for name, p in raw_model.named_parameters():
        if use_muon:
            if name.endswith("qkv_proj.weight"):
                qkv_params.append(p)
                continue
            if ".self_attention" in name or ".feed_forward" in name:
                muon_params.append(p)
                continue
        adamw_params_by_lr_scale.setdefault(hidden_lr_scale(name, p), []).append(p)

    adamw_param_groups = [
        {"params": params, "lr": lr * lr_scale}
        for lr_scale, params in adamw_params_by_lr_scale.items()
    ]
    adamw_optimizer = AdamW(
        adamw_param_groups,
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        eps=1e-10,
        fused=True,
    )
    if not use_muon:
        return adamw_optimizer

    muon_param_groups = []
    muon_param_groups.append({
        'params': qkv_params,
        'param_split_fn': qkv_split_fn,
        'param_recombine_fn': qkv_recombine_fn,
    })
    muon_param_groups.append({
        'params': muon_params,
    })
    return Muon(
        params=muon_param_groups,
        scalar_optimizer=adamw_optimizer,
        lr=muon_lr,
        weight_decay=0.0,
        adjust_lr='spectral_norm',  # https://jeremybernste.in/writing/deriving-muon
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        help="Number of training steps",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=256,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
        default=4e-4,
    )
    parser.add_argument(
        "--muon_lr",
        type=float,
        help="Muon learning rate",
        default=4e-3,
    )
    parser.add_argument(
        "--muon",
        action="store_true",
        help="Use Muon optimizer",
    )
    parser.add_argument(
        "--heads",
        type=int,
        help="Number of attention heads",
        default=8,
    )
    parser.add_argument(
        "--layers",
        type=int,
        help="Number of layers",
        default=24,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed",
        default=0,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Local directory for checkpoint files",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a local checkpoint .pt file to resume training from",
    )
    args = parser.parse_args()
    steps = args.steps
    batch_size = args.batch_size

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{ddp_local_rank}")
        init_process_group(backend="nccl", device_id=device)
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        # vanially non-DDP training
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        print(f"Using device: {device}")

    # Seed differently per rank for better diversity in DDP
    torch.manual_seed(args.seed + ddp_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + ddp_rank)
    np.random.seed(args.seed + ddp_rank)
    random.seed(args.seed + ddp_rank)

    torch.backends.cudnn.benchmark = True

    dataset_id = "benjamin-paine/imagenet-1k-256x256"
    print(f"Loading dataset from {dataset_id}...")
    train_dataset = load_dataset(dataset_id, split="train", num_proc=8)
    val_dataset = load_dataset(dataset_id, split="validation", num_proc=8)
    train_dataset.set_transform(HFImageTransform(random_flip=True))
    val_dataset.set_transform(HFImageTransform())

    num_workers = min(16, multiprocessing.cpu_count() // ddp_world_size)
    if ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True, drop_last=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size // ddp_world_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size // ddp_world_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4
        )
        barrier()
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=4
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=4
        )

    model = create_model(n_heads=args.heads, n_layers=args.layers).to(device)
    raw_model = model
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=0.9999))

    if master_process:
        run = wandb.init(
            project="dit-muon",
            config={
                "d_model": model.config.d_model,
                "d_ff": model.config.d_ff,
                "n_heads": model.config.n_heads,
                "n_layers": model.config.n_layers,
                "image_size": model.config.image_size,
                "patch_size": model.config.patch_size,
                "batch_size": batch_size,
                "steps": steps,
                "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "lr": args.lr,
                "muon_lr": args.muon_lr if args.muon else None,
                "optimizer": "muon" if args.muon else "adamw",
                "seed": args.seed,
                "resume_checkpoint": args.resume_checkpoint,
            }
        )

    optimizer = setup_optimizer(raw_model, args.muon, args.lr, args.muon_lr, ddp)

    start_step = 0
    epoch = 0
    timed_training_seconds = 0.0
    if args.resume_checkpoint is not None:
        if not os.path.isfile(args.resume_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_checkpoint}")
        start_step, epoch, timed_training_seconds = load_checkpoint(
            args.resume_checkpoint,
            raw_model,
            ema_model,
            optimizer,
            device,
        )
        if start_step >= steps:
            raise ValueError(
                f"Checkpoint step ({start_step}) must be smaller than requested --steps ({steps})."
            )
        if master_process:
            print(f"Resumed training from {args.resume_checkpoint} at step {start_step}, epoch {epoch}")

    model = torch.compile(model, backend="inductor", mode="max-autotune")
    ema_model = torch.compile(ema_model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        train_loader.sampler.set_epoch(epoch)

    train_loader_iterator = iter(train_loader)
    # Fast-forward dataloader iterator to approximately preserve where training left off.
    steps_per_epoch = len(train_loader)
    if steps_per_epoch > 0:
        resume_offset = start_step % steps_per_epoch
        for _ in range(resume_offset):
            try:
                next(train_loader_iterator)
            except StopIteration:
                epoch += 1
                if ddp:
                    train_loader.sampler.set_epoch(epoch)
                train_loader_iterator = iter(train_loader)
                next(train_loader_iterator)

    model.train()
    ema_model.eval()
    x_pos = generate_x_pos(raw_model.config.image_size, raw_model.config.patch_size, device)
    training_timer_started_at = time.perf_counter() if start_step >= 4 else None

    for step in tqdm(range(start_step, steps), initial=start_step, total=steps, disable=not master_process):
        try:
            train_batch = next(train_loader_iterator)
        except StopIteration:
            epoch += 1
            if ddp:
                train_loader.sampler.set_epoch(epoch)
            train_loader_iterator = iter(train_loader)
            train_batch = next(train_loader_iterator)

        train_images = images_uint8_to_device_float(train_batch["image"], device, normalize=True)
        train_classes = train_batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            B = train_images.shape[0]
            t = sample_timestep_logit_normal(B, device, mean=-0.8, std=0.8)
            noise = torch.randn_like(train_images)

            _t = t.view(-1, 1, 1, 1)
            x_t = train_images * _t + noise * (1 - _t)
            denominator = (1 - _t).clamp(min=0.05)
            v_target = (train_images - x_t) / denominator
            x_pred = model(x_t, x_pos, t, train_classes)
            v_pred = (x_pred - x_t) / denominator
            loss = F.mse_loss(v_pred, v_target)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=float('inf'))
        optimizer.step()

        if ddp:
            # Average losses across all GPUs for logging
            all_reduce(loss, op=ReduceOp.AVG)

        # Start EMA at 20k steps: first call copies weights, later calls apply EMA decay.
        if step >= 20_000:
            ema_model.update_parameters(raw_model)

        if step + 1 == 4 and training_timer_started_at is None:
            # start training timer after 4 steps to warm up the model
            training_timer_started_at = time.perf_counter()

        # validate at the end of the training or every 50k steps
        is_val_step = (step + 1) % 50_000 == 0 or step + 1 == steps
        
        if master_process:
            should_commit = step % 32 == 0
            run.log(
                {"loss": loss.item(), "grad_norm": grad_norm.item()},
                step=step + 1,
                commit=should_commit and not is_val_step,
            )

        if is_val_step:
            if training_timer_started_at is not None:
                timed_training_seconds += time.perf_counter() - training_timer_started_at

            if master_process:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    args.checkpoint_dir, f"checkpoint_step_{step + 1}.pt"
                )
                save_checkpoint(
                    checkpoint_path,
                    step + 1,
                    epoch,
                    raw_model,
                    ema_model,
                    optimizer,
                    args,
                    timed_training_seconds,
                )

            if ddp:
                barrier()

            fid_score = compute_fid(
                ema_model,
                device,
                val_loader,
                master_process=master_process,
                rank=ddp_rank,
            )

            if master_process:
                run.log({"fid": fid_score, "timed_training_seconds": timed_training_seconds}, step=step + 1, commit=False)
                # Generate and log 64 images
                samples = sample_images(
                    ema_model,
                    device,
                    n=64,
                    seed=0,
                )
                images_grid = torchvision.utils.make_grid(samples, nrow=8, padding=0).cpu()
                run.log({"generated_images": wandb.Image(images_grid)}, step=step + 1, commit=True)
        
            # Ensure all processes wait for validation to complete before continuing
            if ddp:
                barrier()

            if training_timer_started_at is not None:
                training_timer_started_at = time.perf_counter()

    if master_process:
        # Save EMA model to Wandb
        torch.save(ema_model._orig_mod.module.state_dict(), "model.pth")
        artifact = wandb.Artifact("imagenet-256-dit-ema", type="model")
        artifact.add_file("model.pth")
        run.log_artifact(artifact)
        run.finish()
    
    if ddp:
        barrier()  # Sync all processes before cleanup
        destroy_process_group()


if __name__ == "__main__":
    main()