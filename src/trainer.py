import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from src.lr_schedule import get_lr
from src.checkpoint import save_checkpoint


def pick_precision(device, requested):
    if device.type != "cuda":
        if requested not in ("auto", "fp32"):
            print("No CUDA device - ignoring precision={!r}, using fp32.".format(requested))
        return torch.float32

    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if requested == "fp32":
        return torch.float32

    # auto
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class Trainer:
    def __init__(self, model, dataset_builder, train_bin, val_bin,
                 model_config, train_config, device):
        self.model = model.to(device)
        self.dataset_builder = dataset_builder  # self supervised learning
        self.train_bin = train_bin
        self.val_bin = val_bin
        self.model_config = model_config
        self.train_config = train_config
        self.device = device

        self.dtype = pick_precision(device, train_config.precision)
        self.use_amp = device.type == "cuda" and self.dtype != torch.float32
        use_fp16 = self.dtype == torch.float16
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
        except (AttributeError, TypeError):
            # torch < 2.3 does not have the unified torch.amp.GradScaler.
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

        self.optimizer = model.configure_optimizers(
            weight_decay=train_config.weight_decay,
            lr=train_config.lr_max,
            betas=(train_config.beta1, train_config.beta2),
        )

        print("Device:", device, "| precision:", self.dtype)

    def _get_batch(self, split):
        path = self.train_bin if split == "train" else self.val_bin
        x, y = self.dataset_builder.get_batch(
            path, self.train_config.batch_size, self.model_config.block_size
        )
        x = torch.from_numpy(x).to(self.device, non_blocking=True)
        y = torch.from_numpy(y).to(self.device, non_blocking=True)
        return x, y

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(self.train_config.eval_iters)
            for i in range(self.train_config.eval_iters):
                x, y = self._get_batch(split)
                with self._autocast():
                    _, loss = self.model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def train(self, start_step=0, best_val_loss=float("inf"), log_path=None):
        cfg = self.train_config
        log_path = Path(log_path) if log_path else None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.train()
        t_last_log = time.time()
        tokens_per_step = cfg.tokens_per_step(self.model_config.block_size)

        print("Tokens per optimizer step:", tokens_per_step)
        print("Planned total tokens     :", tokens_per_step * cfg.max_steps)
        print()

        for step in range(start_step, cfg.max_steps):
            lr = get_lr(step, cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                x, y = self._get_batch("train")
                with self._autocast():
                    _, loss = self.model(x, y)
                    loss = loss / cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if step % cfg.log_interval == 0 or step == cfg.max_steps - 1:
                now = time.time()
                elapsed = now - t_last_log
                toks_per_sec = (tokens_per_step * cfg.log_interval) / max(elapsed, 1e-6)
                t_last_log = now
                eta_min = (cfg.max_steps - step) * tokens_per_step / max(toks_per_sec, 1e-6) / 60

                record = {
                    "step": step,
                    "train_loss": accum_loss,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tokens_per_sec": toks_per_sec,
                }
                print(
                    "step {:5d} | loss {:.4f} | lr {:.2e} | grad_norm {:.2f} "
                    "| tok/s {:,.0f} | eta {:.0f} min".format(
                        step, accum_loss, lr, float(grad_norm), toks_per_sec, eta_min
                    )
                )
                if log_path:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")

            do_eval = step % cfg.eval_interval == 0 and step > start_step
            do_ckpt = step % cfg.checkpoint_interval == 0 and step > start_step
            is_last = step == cfg.max_steps - 1

            if do_eval or is_last:
                losses = self.estimate_loss()
                print(
                    "  eval @ step {:5d} | train {:.4f} | val {:.4f}".format(
                        step, losses["train"], losses["val"]
                    )
                )
                if log_path:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, "eval": losses}) + "\n")

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    best_path = Path(cfg.out_dir) / "best.pt"
                    save_checkpoint(
                        best_path, self.model, self.optimizer,
                        step, best_val_loss, self.model_config, cfg,
                    )
                    print("  -> new best val loss, saved", best_path)

            if do_ckpt or is_last:
                save_checkpoint(
                    Path(cfg.out_dir) / "last.pt", self.model, self.optimizer,
                    step, best_val_loss, self.model_config, cfg,
                )

        return best_val_loss