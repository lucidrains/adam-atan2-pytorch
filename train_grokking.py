# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "adam-atan2-pytorch>=0.3.4",
#     "tqdm",
#     "x-transformers",
# ]
# ///

import torch
import torch.nn.functional as F

from x_transformers import TransformerWrapper, Decoder
from adam_atan2_pytorch import AdamAtan2ClipGrok

from tqdm import tqdm

# helpers

def mod_inverse(a, p):
    return pow(a, p - 2, p)

def get_batch(data, batch_size):
    idx = torch.randint(0, len(data), (batch_size,))
    seq = data[idx]
    return seq[:, :-1], seq[:, -1]

# main

def main(
    batch_size = 512,
    max_steps = 50_000,
    learning_rate = 1e-3,
    weight_decay = 1e-1,
    eval_every = 100,
    grok_threshold = 0.95,
    prime = 97,
    fraction_train = 0.25
):
    # dataset - modular division a * b^-1 mod prime

    data = []

    for a in range(prime):
        for b in range(1, prime):
            c = (a * mod_inverse(b, prime)) % prime
            data.append([a, prime, b, prime + 1, c])

    data = torch.tensor(data)

    # train / val split

    torch.manual_seed(42)
    indices = torch.randperm(len(data))
    n_train = int(len(data) * fraction_train)

    train_data, val_data = data[indices[:n_train]], data[indices[n_train:]]

    # device

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")

    train_data, val_data = train_data.to(device), val_data.to(device)

    # training loop

    def train(use_clip_to_grok = False):
        torch.manual_seed(42)

        model = TransformerWrapper(
            num_tokens = prime + 2,
            max_seq_len = 5,
            attn_layers = Decoder(
                dim = 128,
                depth = 2,
                heads = 4,
                use_rmsnorm = True
            )
        ).to(device)

        optimizer = AdamAtan2ClipGrok(
            model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay,
            clip_to_grok = use_clip_to_grok
        )

        pbar = tqdm(range(max_steps))
        best_acc = 0.0

        for step in pbar:
            model.train()

            x, y = get_batch(train_data, batch_size)
            logits = model(x)
            loss = F.cross_entropy(logits[:, -1], y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (step + 1) % eval_every == 0:
                model.eval()

                with torch.no_grad():
                    val_x, val_y = val_data[:, :-1], val_data[:, -1]
                    val_logits = model(val_x)
                    acc = (val_logits[:, -1].argmax(dim = -1) == val_y).float().mean().item()

                best_acc = max(best_acc, acc)
                pbar.set_description(f"loss: {loss.item():.4f} | acc: {acc:.3f} | best: {best_acc:.3f}")

                if acc > grok_threshold:
                    return step + 1

        return max_steps

    # compare baseline vs clip to grok

    print("\n--- Training AdamAtan2 (baseline / clip disabled) ---")
    steps_baseline = train(use_clip_to_grok = False)
    print(f"\nAdamAtan2ClipGrok (baseline) reached grokking (>{grok_threshold:.0%} val acc) in {steps_baseline} steps.")

    print("\n--- Training AdamAtan2ClipGrok (clip enabled) ---")
    steps_clip = train(use_clip_to_grok = True)
    print(f"\nAdamAtan2ClipGrok (clip enabled) reached grokking (>{grok_threshold:.0%} val acc) in {steps_clip} steps.")

    print(f"\n{'=' * 50}")
    print(f"AdamAtan2 (baseline):         {steps_baseline} steps")
    print(f"AdamAtan2ClipGrok (clip):     {steps_clip} steps")

    if steps_clip < steps_baseline:
        print(f"Speedup:                      {steps_baseline / steps_clip:.1f}x faster with ClipToGrok")

if __name__ == '__main__':
    main()
