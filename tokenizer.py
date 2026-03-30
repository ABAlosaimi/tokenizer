from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tokenizers.normalizers import NFKC
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.Sequence([
    NFKC(),  # Normalization Form Compatibility Composition
])

tokenizer.pre_tokenizer = Whitespace()

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = BpeTrainer(vocab_size=30000, special_tokens=special_tokens)

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

encoding = tokenizer.encode("Hello, how are you doing today?")
print("Tokens:", encoding.tokens)