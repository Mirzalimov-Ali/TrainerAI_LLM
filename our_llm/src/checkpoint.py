import os
import tempfile
from pathlib import Path

import torch

def save_checkpoint(path, model, optimizer, step, best_val_loss, model_config, train_config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "best_val_loss": best_val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": vars(train_config),
    }

    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)  # atomic on both POSIX and Windows
    except Exception:
        os.unlink(tmp_path)
        raise


def load_checkpoint(path, model, optimizer=None, map_location=None):
    payload = torch.load(path, map_location=map_location, weights_only=True)

    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    return payload["step"], payload["best_val_loss"]