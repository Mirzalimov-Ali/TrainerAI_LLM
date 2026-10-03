from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, pre_tokenizers, decoders, trainers

SPECIAL_TOKENS = ["<eos>"]

class Tokenizer:
    def __init__(self):
        self.tokenizer = None

    def train(self, texts, vocab_size=8000, min_frequency=2):
        tokenizer = HFTokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            min_frequency=min_frequency,
            show_progress=True,
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)
        self.tokenizer = tokenizer

        print("Trained tokenizer. Vocabulary size:", self.vocab_size())
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))
        print("Saved tokenizer to", path)

    def load(self, path):
        self.tokenizer = HFTokenizer.from_file(str(path))
        return self

    def encode(self, text):
        """text -> list of token IDs"""
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        """list of token IDs -> text"""
        return self.tokenizer.decode(ids)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def eos_id(self):
        return self.tokenizer.token_to_id("<eos>")