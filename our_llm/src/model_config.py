from dataclasses import dataclass, asdict
import yaml

@dataclass
class ModelConfig:
    vocab_size: int = 8000
    block_size: int = 256
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 6
    dropout: float = 0.1
    ffn_mult: int = 4
    tie_weights: bool = True

    def head_dim(self):
        return self.n_embd // self.n_head

    def check(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                "n_embd ({}) must divide evenly by n_head ({})".format(
                    self.n_embd, self.n_head)
            )
        if self.vocab_size > 65536:
            raise ValueError(
                "vocab_size above 65536 does not fit in the uint16 token files"
            )

    def estimate_parameters(self):
        non_embedding = 12 * self.n_layer * self.n_embd ** 2
        embedding = self.vocab_size * self.n_embd
        position = self.block_size * self.n_embd   # learned position embeddings

        total = non_embedding + embedding + position
        if not self.tie_weights:
            total += self.vocab_size * self.n_embd   # a separate output layer

        return {
            "non_embedding": non_embedding,
            "embedding": embedding,
            "position": position,
            "total": total,
        }

    def estimate_training_memory_mb(self):
        total = self.estimate_parameters()["total"]
        return total * 16 / (1024 ** 2)

    def to_dict(self):
        return asdict(self)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f)

    config = ModelConfig(**values)
    config.check()
    return config