from model import GPTLanguageModel, GPTConfig
import torch

vocab_size = 30000  # from tokenizer
block_size = 128

config = GPTConfig(vocab_size, block_size)
model = GPTLanguageModel(config)

x = torch.randint(0, vocab_size, (1, 32))  # fake input
logits, loss = model(x, x)
print("✅ Forward pass works! Loss =", loss.item())


