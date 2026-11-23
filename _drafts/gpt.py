"""
gpt.py - a GPT-2 style language model in a single file, trained on TinyShakespeare.

Companion code for the article "From Linear Regression to GPT-2".
Each "Step" section below matches a section of the article. The architecture
follows GPT-2; only the sizes are scaled down so it trains on a laptop CPU.
At every step the comments show GPT-2's real size next to the size we use.

Usage: set MODE below, then run `python gpt.py`.

Inspired by Andrej Karpathy's "Let's build GPT" lecture (ng-video-lecture).
"""

import math
import os
import time
import urllib.request

import torch

# Rule for this file: PyTorch only gives us tensors and autograd (gradients).
# Every layer, the matrix multiplications and the weight updates are written by hand.

# -----------------------------------------------------------------------------
# Switch: what running this file does
#   "train":    train the model, saving a checkpoint as it goes, then generate a sample
#   "generate": load the saved checkpoint and continue the prompt (no training)
MODE = "generate"

# -----------------------------------------------------------------------------
# Model hyperparameters                               GPT-2 (124M)   Ours
batch_size = 32     # sequences per training step            512        32
block_size = 128    # context length, in tokens             1024       128
n_embd = 256        # size of each token's vector            768       256
n_layer = 6         # transformer blocks                      12         6
n_head = 4          # attention heads per block               12         4
head_size = n_embd // n_head  # size of each head's output     64        64

# Training hyperparameters
#   The GPT-2 paper doesn't publish these. The values follow the conventions of
#   GPT-3 and nanoGPT, sized for a ~1-2 hour run on a laptop CPU.
max_iters = 1500        # training steps (~4.5 s/step on a 4-core laptop CPU: ~2 hours)
eval_interval = 250     # measure train/val loss (and save a checkpoint) every N steps
eval_iters = 20         # batches averaged per loss measurement
learning_rate = 1e-3    # peak learning rate, reached after warmup
min_lr = 1e-4           # learning rate at the end of the cosine decay
warmup_iters = 100      # steps to ramp the learning rate up from ~0
weight_decay = 0.1      # AdamW: pull weight matrices slightly towards 0 each step
beta1, beta2 = 0.9, 0.99  # AdamW: how slowly the gradient averages forget the past
grad_clip = 1.0         # cap on the total gradient norm
resume = False          # True: continue from the last checkpoint instead of starting over

CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# -----------------------------------------------------------------------------
# Data: TinyShakespeare (~1 MB, all of Shakespeare's plays in one text file)
#   GPT-2 was trained on WebText: ~40 GB of text from 8M web pages.
#   We use ~1 MB. That difference matters more than any architectural detail.

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")

if not os.path.exists(DATA_PATH):
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH, encoding="utf-8") as f:
    text = f.read()

# -----------------------------------------------------------------------------
# Step 1: Tokenization - text -> list of integers
#   GPT-2: byte-level BPE, vocab_size = 50,257 (sub-word pieces like "machines", " think")
#   Ours:  character-level, vocab_size = 65 (every distinct character in the dataset)
#   1 MB of text is far too little to train 50,257 embeddings, so we use characters.

chars = sorted(set(text))                       # the vocabulary: every distinct character
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}    # string -> integer
itos = {i: ch for i, ch in enumerate(chars)}    # integer -> string


def encode(s):
    """Text -> token ids, e.g. "Can" -> [15, 39, 52]."""
    return [stoi[c] for c in s]


def decode(ids):
    """Token ids -> text, e.g. [15, 39, 52] -> "Can"."""
    return "".join(itos[i] for i in ids)


# Encode the whole dataset once, then hold out the last 10% to measure
# how well the model does on text it never trained on.
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """Pick batch_size random chunks of block_size tokens.

    x is the input, y is the same chunk shifted one token to the left:
    y[b, t] is the token that comes right after x[b, t], i.e. what the model
    should predict at position t.
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))   # random start positions
    x = torch.stack([data[i : i + block_size] for i in ix])          # (B, T)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # (B, T)
    return x.to(device), y.to(device)


# -----------------------------------------------------------------------------
# Building blocks

def new_param(shape, std=0.02):
    """A learnable weight tensor: small random values, tracked by autograd."""
    return (torch.randn(shape, device=device) * std).requires_grad_()


def new_const_param(shape, value):
    """A learnable tensor that starts as a constant (biases and LayerNorm beta at 0, LayerNorm gamma at 1)."""
    return torch.full(shape, value, device=device).requires_grad_()


# -----------------------------------------------------------------------------
# The model. Shapes in comments use:
#   B = batch size, T = sequence length (<= block_size), C = n_embd

class GPT:

    def __init__(self):
        # All learnable weights live in one dict, so the training loop can find,
        # update and save them. The names follow OpenAI's GPT-2 weights, except
        # attention, where we give every head its own Q, K and V.
        self.params = {}

        p = self.params
        C = n_embd

        # Step 2: Token and position embeddings
        # Both are lookup tables: row i is a learned vector of n_embd numbers.
        # GPT-2 starts them with small random values (std 0.02 and 0.01).
        #                                                 GPT-2            Ours
        p["wte"] = new_param((vocab_size, C), std=0.02)  # (50,257 x 768)   (65 x 256)   one row per token
        p["wpe"] = new_param((block_size, C), std=0.01)  # ( 1,024 x 768)   (128 x 256)  one row per position

        # Steps 3-6: n_layer transformer blocks, each with its own weights.
        # GPT-2 starts the two projections that write into the residual stream
        # smaller, so the stream doesn't grow as blocks keep adding to it.
        proj_std = 0.02 / (2 * n_layer) ** 0.5
        for l in range(n_layer):
            h = f"h.{l}."
            # Step 3: LayerNorm before attention: a scale and a shift per channel
            p[h + "ln_1.gamma"] = new_const_param((C,), 1.0)                # (C,)  scale
            p[h + "ln_1.beta"] = new_const_param((C,), 0.0)                 # (C,)  shift
            # Step 4: Attention heads: every head has its own query, key and value
            # matrices, each projecting a token's C values down to head_size.
            # (GPT-2 stores all heads' Q, K and V side by side in one (C, 3C)
            # matrix called c_attn; the math is the same.)
            for i in range(n_head):
                a = h + f"attn.head.{i}."
                p[a + "q.weight"] = new_param((C, head_size))               # (C, hs)
                p[a + "q.bias"] = new_const_param((head_size,), 0.0)        # (hs,)
                p[a + "k.weight"] = new_param((C, head_size))               # (C, hs)
                p[a + "k.bias"] = new_const_param((head_size,), 0.0)        # (hs,)
                p[a + "v.weight"] = new_param((C, head_size))               # (C, hs)
                p[a + "v.bias"] = new_const_param((head_size,), 0.0)        # (hs,)
            # Step 5: Output projection: mixes the heads' outputs back into the stream
            p[h + "attn.c_proj.weight"] = new_param((C, C), std=proj_std)   # (C, C)
            p[h + "attn.c_proj.bias"] = new_const_param((C,), 0.0)          # (C,)
            # Step 3 again: LayerNorm before the MLP
            p[h + "ln_2.gamma"] = new_const_param((C,), 1.0)                # (C,)  scale
            p[h + "ln_2.beta"] = new_const_param((C,), 0.0)                 # (C,)  shift
            # Step 6: MLP: expand to 4C, then project back to C
            p[h + "mlp.c_fc.weight"] = new_param((C, 4 * C))                # (C, 4C)
            p[h + "mlp.c_fc.bias"] = new_const_param((4 * C,), 0.0)         # (4C,)
            p[h + "mlp.c_proj.weight"] = new_param((4 * C, C), std=proj_std) # (4C, C)
            p[h + "mlp.c_proj.bias"] = new_const_param((C,), 0.0)           # (C,)

        # Final LayerNorm, after the last block
        p["ln_f.gamma"] = new_const_param((C,), 1.0)
        p["ln_f.beta"] = new_const_param((C,), 0.0)

        # Step 7: Unembedding has no weights of its own. GPT-2 reuses wte
        # ("weight tying"): the same table that turns tokens into vectors turns
        # vectors back into a score per token.

    def parameters(self):
        return list(self.params.values())

    def forward(self, idx):
        # idx: token ids, shape (B, T)
        B, T = idx.shape
        p = self.params

        # Step 2: Embeddings. Their sum is the start of the residual stream x.
        tok_emb = p["wte"][idx]                    # (B, T, C)  row lookup: what each token is
        pos_emb = p["wpe"][:T]                     # (T, C)     first T rows: where each token is
        value_in_residual_stream = tok_emb + pos_emb                      # (B, T, C)  broadcast: same positions for every sequence

        # The residual stream x keeps its shape (B, T, C) all the way through.
        # Each block reads from it, computes an update, and adds the update back.
        for layer in range(n_layer):

            # Attention: tokens look at earlier tokens and gather information
            h = self.layer_norm(value_in_residual_stream, f"h.{layer}.ln_1")          # (B, T, C)          Step 3
            heads = []
            for head in range(n_head):
                heads.append(self.attention_head(h, layer, head))  # (B, T, head_size) Step 4
            a = torch.cat(heads, dim=-1)                       # (B, T, C)          heads side by side
            a = self.output_projection(a, layer)               # (B, T, C)          Step 5
            value_in_residual_stream = value_in_residual_stream + a                                          # residual add

            # MLP: each token thinks on its own about what it gathered
            h = self.layer_norm(value_in_residual_stream, f"h.{layer}.ln_2")          # (B, T, C)          Step 3
            m = self.mlp(h, layer)                             # (B, T, C)          Step 6
            value_in_residual_stream = value_in_residual_stream + m                                          # residual add

        value_in_residual_stream = self.layer_norm(value_in_residual_stream, "ln_f")                         # (B, T, C)
        logits = self.unembed(value_in_residual_stream)                               # (B, T, vocab_size) Step 7

        # logits[b, t] holds one score per token in the vocabulary: the model's
        # prediction for the token that comes after position t.
        return logits

    def generate(self, idx, max_new_tokens):
        """Inference: extend idx (B, T) by max_new_tokens tokens, one at a time."""
        with torch.no_grad():                      # no gradients needed for inference
            for _ in range(max_new_tokens):
                context = idx[:, -block_size:]     # (B, <=block_size)  the model only sees the last block_size tokens
                logits = self.forward(context)     # (B, T, vocab_size)
                logits = logits[:, -1, :]          # (B, vocab_size)    only the last position predicts the next token
                probs = self.softmax(logits)       # (B, vocab_size)    Step 8: scores -> probabilities
                next_id = self.sample(probs)       # (B, 1)             Step 9: pick one token
                idx = torch.cat([idx, next_id], dim=1)  # (B, T+1)      append it and repeat
        return idx

    # -------------------------------------------------------------------------
    # Step 3: LayerNorm
    #   Each block keeps adding to the residual stream, so the values in x can
    #   drift to any scale. Before attention and the MLP read from the stream,
    #   LayerNorm rescales each token's vector to mean 0 and variance 1, so they
    #   always get input in the same range. Then a learned scale (gamma) and
    #   shift (beta) per channel let the model undo that where it helps:
    #   y = x_hat * gamma + beta, element by element (not a dot product).
    #   Each token is normalized on its own, across its C channels: tokens and
    #   sequences never mix here.
    #                                                GPT-2      Ours
    #   channels normalized per token (C)              768       256
    #   learned weights per LayerNorm (scale + shift)  1,536     512
    #   LayerNorms in the model (2 per block + final)  25        13

    def layer_norm(self, x, name, eps=1e-5):
        """Normalize each token's vector, then scale and shift it.  (B, T, C) -> (B, T, C)"""
        gamma = self.params[name + ".gamma"]                   # (C,)  scale, starts at 1
        beta = self.params[name + ".beta"]                     # (C,)  shift, starts at 0

        mean = x.mean(dim=-1, keepdim=True)                    # (B, T, 1)  average of each token's C values
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)     # (B, T, 1)  their spread
        x_hat = (x - mean) / torch.sqrt(var + eps)             # (B, T, C)  mean 0, variance 1; eps avoids dividing by 0
        return x_hat * gamma + beta                            # (B, T, C)  learned scale and shift per channel

    # -------------------------------------------------------------------------
    # Step 4: Attention head (causal self-attention)
    #   Attention is how tokens share information. Every token asks a question
    #   (query), advertises what it contains (key), and offers what it would
    #   pass on (value). A token's output is a weighted average of the values
    #   of the tokens it attends to, weighted by how well its query matches
    #   their keys.
    #   "Causal": a token may only look at itself and earlier tokens. When
    #   predicting the next character, the model must not peek at it.
    #   Every head has its own Wq, Wk and Wv, each (C x head_size): it reads
    #   the full token vector and projects it down to 64 values. With separate
    #   weights, the heads can learn to look for different things.
    #   All heads together: n_head x head_size = C (4 x 64 = 256).
    #                                                GPT-2      Ours
    #   heads per block                                 12         4
    #   Wq, Wk, Wv per head (C x head_size each)    768x64    256x64
    #   size of each head's q, k, v (head_size)         64        64
    #   attention scores per head (T x T)        1024x1024   128x128

    def attention_head(self, x, layer, head):
        """One causal self-attention head.  (B, T, C) -> (B, T, head_size)"""
        B, T, C = x.shape
        p = self.params
        a = f"h.{layer}.attn.head.{head}."                     # this head's own weights

        q = x @ p[a + "q.weight"] + p[a + "q.bias"]            # (B, T, C) @ (C, hs) -> (B, T, hs)  what each token is looking for
        k = x @ p[a + "k.weight"] + p[a + "k.bias"]            # (B, T, hs)  what each token contains
        v = x @ p[a + "v.weight"] + p[a + "v.bias"]            # (B, T, hs)  what each token passes on

        # How well each query matches each key. Dividing by sqrt(head_size)
        # keeps the scores from growing with head_size, which would make the
        # softmax too peaked and its gradients tiny.
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)  # (B, T, T)  scores[b, i, j]: token i looking at token j

        # Causal mask: token i may not look at tokens j > i (the future).
        # Setting those scores to -inf makes their softmax weight exactly 0.
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))  # lower triangle = allowed
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = self.softmax(scores)                         # (B, T, T)  each row sums to 1
        return weights @ v                                     # (B, T, hs)  weighted average of the values

    # -------------------------------------------------------------------------
    # Step 5: Output projection
    #   The heads' outputs were concatenated side by side: (B, T, n_head * hs).
    #   Each head only produced its own 64 values. This matrix mixes all heads
    #   together and turns the result into an update for the residual stream.
    #                                                GPT-2      Ours
    #   c_proj weight (C x C)                      768x768   256x256

    def output_projection(self, a, layer):
        """Mix the concatenated heads into one update.  (B, T, C) -> (B, T, C)"""
        W = self.params[f"h.{layer}.attn.c_proj.weight"]       # (C, C)
        b = self.params[f"h.{layer}.attn.c_proj.bias"]         # (C,)
        return a @ W + b                                       # (B, T, C)

    # -------------------------------------------------------------------------
    # Step 6: MLP (feed-forward network)
    #   Attention moved information between tokens. The MLP then works on each
    #   token on its own: expand its vector to 4x the size, apply a non-linear
    #   function (GELU), and project back down. Most of the model's parameters
    #   are here, and much of what it "knows" is thought to be stored here.
    #                                                GPT-2      Ours
    #   hidden size (4 x C)                          3,072     1,024
    #   parameters per MLP                           4.7M      0.5M

    def gelu(self, x):
        """GELU, in the tanh approximation GPT-2 uses.

        Like ReLU (keeps positive values, drops negative ones) but smooth:
        small negative values are let through a little.
        """
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

    def mlp(self, x, layer):
        """Expand, GELU, project back.  (B, T, C) -> (B, T, C)"""
        p = self.params
        h = x @ p[f"h.{layer}.mlp.c_fc.weight"] + p[f"h.{layer}.mlp.c_fc.bias"]        # (B, T, 4C)  expand
        h = self.gelu(h)                                                               # (B, T, 4C)  non-linearity
        return h @ p[f"h.{layer}.mlp.c_proj.weight"] + p[f"h.{layer}.mlp.c_proj.bias"]  # (B, T, C)   project back

    # -------------------------------------------------------------------------
    # Step 7: Unembedding
    #   Turn each token's final vector into one score (logit) per vocabulary
    #   token. GPT-2 has no separate matrix for this: it reuses the token
    #   embedding table wte ("weight tying"). The score for token v is the dot
    #   product of x with row v of wte, i.e. how similar x is to token v's
    #   embedding.
    #                                                GPT-2         Ours
    #   scores per position (vocab_size)              50,257          65
    #   extra parameters (thanks to weight tying)          0           0

    def unembed(self, x):
        """One score per vocabulary token.  (B, T, C) -> (B, T, vocab_size)"""
        return x @ self.params["wte"].T                        # (B, T, C) @ (C, V) -> (B, T, V)

    # -------------------------------------------------------------------------
    # Step 8: Softmax
    #   Turns any list of scores into probabilities: all positive, summing to 1,
    #   with higher scores getting more probability. exp(-inf) = 0, which is
    #   what makes the attention mask work. Subtracting the max first doesn't
    #   change the result, but keeps exp() from overflowing.

    def softmax(self, x):
        """Softmax over the last dimension.  (..., N) -> (..., N)"""
        e = torch.exp(x - x.max(dim=-1, keepdim=True).values)
        return e / e.sum(dim=-1, keepdim=True)

    # -------------------------------------------------------------------------
    # Step 9: Sampling
    #   Pick the next token at random, in proportion to its probability.
    #   Picking the most likely token every time (greedy) makes text repetitive;
    #   sampling keeps it varied.
    #   How: line the probabilities up end to end from 0 to 1 (cumulative sum),
    #   throw a random number r in [0, 1), and take the token whose stretch it
    #   lands in.

    def sample(self, probs):
        """Pick one token id per row according to probs.  (B, vocab_size) -> (B, 1)"""
        B, V = probs.shape
        cdf = probs.cumsum(dim=-1)                             # (B, V)  e.g. [0.1, 0.3, 0.9, 1.0]
        r = torch.rand(B, 1, device=probs.device)              # (B, 1)  e.g. 0.5
        idx = (cdf < r).sum(dim=-1, keepdim=True)              # (B, 1)  tokens fully below r -> e.g. 2
        return idx.clamp(max=V - 1)                            # guard against rounding: cdf may end at 0.99999


# -----------------------------------------------------------------------------
# Step 10: Loss - how wrong are the predictions?
#   Cross-entropy: the average of -log(probability the model gave the correct
#   next token). An untrained model guessing uniformly over 65 characters
#   scores -log(1/65) = ln(65) = 4.17. Perfect predictions would score 0.

def cross_entropy(logits, targets):
    """logits (B, T, V), targets (B, T) -> one number."""
    B, T, V = logits.shape
    logits = logits.reshape(B * T, V)               # one row of scores per position
    targets = targets.reshape(B * T)                # the correct next token for each row

    # log of softmax, computed stably: subtract each row's max before exp
    # so exp() never overflows. This doesn't change the result.
    m = logits.max(dim=1, keepdim=True).values
    log_probs = logits - m - torch.log(torch.exp(logits - m).sum(dim=1, keepdim=True))  # (B*T, V)

    correct = log_probs[torch.arange(B * T), targets]  # (B*T,) log-probability of the right answer
    return -correct.mean()


# -----------------------------------------------------------------------------
# Step 11: Gradient clipping - guard against one bad batch
#   If the gradients of all weights together (as one long vector) are longer
#   than max_norm, shrink them all by the same factor. Direction is kept,
#   only the step size is capped.

def clip_gradients(params, max_norm):
    grads = [w.grad for w in params if w.grad is not None]
    total_norm = torch.sqrt(torch.stack([(g * g).sum() for g in grads]).sum())
    if total_norm > max_norm:
        for g in grads:
            g.mul_(max_norm / total_norm)
    return total_norm


# -----------------------------------------------------------------------------
# Step 12: Learning-rate schedule - linear warmup, then cosine decay
#   Warmup: start tiny while the weights are random and gradients are noisy.
#   Cosine decay: take smaller and smaller steps as the model settles.

def get_lr(step):
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)   # 0 -> 1
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


# -----------------------------------------------------------------------------
# Step 13: Optimizer - AdamW, by hand
#   Autograd fills w.grad. This class decides how to change w using it.
#   Plain gradient descent would be:  w -= lr * grad
#   Adam keeps two running averages per weight:
#     m: average gradient           -> a smoothed direction (momentum)
#     v: average squared gradient   -> how noisy/large this weight's gradients are
#   and steps by m / sqrt(v), so every weight moves at a sensible speed.
#   The "W" is decoupled weight decay: shrink weight matrices a little each step.

class AdamW:

    def __init__(self, params, betas, weight_decay, eps=1e-8):
        self.params = params                          # the model's dict: name -> tensor
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.t = 0                                    # number of steps taken
        self.m = {k: torch.zeros_like(w) for k, w in params.items()}
        self.v = {k: torch.zeros_like(w) for k, w in params.items()}

    def zero_grad(self):
        """Clear old gradients. Autograd adds to w.grad, it doesn't overwrite it."""
        for w in self.params.values():
            w.grad = None

    def step(self, lr):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        with torch.no_grad():                         # updating weights is not part of the graph
            for k, w in self.params.items():
                g = w.grad
                if g is None:                         # weight not used in this forward pass
                    continue
                m, v = self.m[k], self.v[k]
                m.mul_(b1).add_((1 - b1) * g)         # m = b1*m + (1-b1)*g
                v.mul_(b2).add_((1 - b2) * g * g)     # v = b2*v + (1-b2)*g^2
                m_hat = m / (1 - b1 ** self.t)        # m and v start at 0, which biases them
                v_hat = v / (1 - b2 ** self.t)        # towards 0 early on; this corrects it
                if w.dim() >= 2:                      # decay matrices only, not biases / LayerNorm
                    w.sub_(lr * self.weight_decay * w)
                w.sub_(lr * m_hat / (v_hat.sqrt() + self.eps))

    def state_dict(self):
        return {"t": self.t,
                "m": {k: t.cpu() for k, t in self.m.items()},
                "v": {k: t.cpu() for k, t in self.v.items()}}

    def load_state_dict(self, state):
        self.t = state["t"]
        for k in self.m:
            self.m[k].copy_(state["m"][k])
            self.v[k].copy_(state["v"][k])


# -----------------------------------------------------------------------------
# Step 14: Evaluation - train loss vs. validation loss
#   Average the loss over eval_iters batches from each split.
#   Both falling: the model is learning. Train falling but val rising: it's
#   memorizing the training text (overfitting).

def estimate_loss(model):
    out = {}
    with torch.no_grad():
        for split in ("train", "val"):
            total = 0.0
            for _ in range(eval_iters):
                x, y = get_batch(split)
                total += cross_entropy(model.forward(x), y).item()
            out[split] = total / eval_iters
    return out


# -----------------------------------------------------------------------------
# Checkpoints - save the weights so training doesn't have to be repeated

def model_config():
    """The sizes a checkpoint must match to be loaded into this model."""
    return {"vocab_size": vocab_size, "block_size": block_size,
            "n_embd": n_embd, "n_layer": n_layer, "n_head": n_head}


def save_checkpoint(model, optimizer, step):
    torch.save({
        "config": model_config(),
        "step": step,                                  # steps completed so far
        "params": {k: w.detach().cpu() for k, w in model.params.items()},
        "optimizer": optimizer.state_dict(),
    }, CKPT_PATH)


def load_checkpoint(model, optimizer=None):
    """Copy saved weights into model (and optimizer state, if given). Returns the step."""
    ckpt = torch.load(CKPT_PATH, map_location=device)
    if ckpt["config"] != model_config():
        raise ValueError(f"checkpoint sizes {ckpt['config']} don't match the model {model_config()}")
    if ckpt["params"].keys() != model.params.keys():
        raise ValueError(f"{CKPT_PATH} was saved by an older version of this code "
                         "(its weight names differ). Retrain with MODE = 'train'.")
    with torch.no_grad():
        for k, w in model.params.items():
            w.copy_(ckpt["params"][k])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


# -----------------------------------------------------------------------------
# Step 15: The training loop
#   Repeat: get a batch -> predict -> measure the loss -> compute gradients
#   (autograd) -> update the weights (AdamW).

def train(model):
    optimizer = AdamW(model.params, betas=(beta1, beta2), weight_decay=weight_decay)

    if not resume and os.path.exists(CKPT_PATH):
        raise SystemExit(f"{CKPT_PATH} already exists. Delete it to train from scratch, "
                         "or set resume = True to continue it.")

    start = 0
    if resume and os.path.exists(CKPT_PATH):
        start = load_checkpoint(model, optimizer)
        print(f"resuming from step {start}")

    t0 = time.time()
    for step in range(start, max_iters):

        # Every eval_interval steps: report progress and save a checkpoint
        if step % eval_interval == 0:
            losses = estimate_loss(model)
            done = step - start
            if done > 0:
                sec_per_step = (time.time() - t0) / done
                eta = sec_per_step * (max_iters - step) / 60
                timing = f" | {sec_per_step * 1000:.0f} ms/step, ~{eta:.0f} min left"
                save_checkpoint(model, optimizer, step)
            else:
                timing = ""
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}"
                  f" | lr {get_lr(step):.2e}{timing}")

        lr = get_lr(step)
        x, y = get_batch("train")                     # (B, T), (B, T)
        logits = model.forward(x)                     # (B, T, vocab_size)
        loss = cross_entropy(logits, y)               # one number

        optimizer.zero_grad()
        loss.backward()                               # autograd: fills w.grad for every weight
        clip_gradients(model.parameters(), grad_clip)
        optimizer.step(lr)                            # our AdamW: changes the weights

    losses = estimate_loss(model)
    print(f"step {max_iters:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | done")
    save_checkpoint(model, optimizer, max_iters)
    print(f"saved checkpoint to {CKPT_PATH}")


# -----------------------------------------------------------------------------
# Step 16: Inference - continue a prompt
#   Text -> token ids -> the model adds one token at a time -> back to text.

def load_trained_weights(model):
    """Load the saved checkpoint into model, if there is one."""
    if not os.path.exists(CKPT_PATH):
        print("no checkpoint found: generating with an untrained model")
        return
    step = load_checkpoint(model)
    print(f"loaded checkpoint trained for {step} steps")


def complete(model, prompt, max_new_tokens):
    """Continue prompt by max_new_tokens characters."""
    idx = torch.tensor([encode(prompt)], device=device)  # (1, T)
    out = model.generate(idx, max_new_tokens)            # (1, T + max_new_tokens)
    return decode(out[0].tolist())


# -----------------------------------------------------------------------------
# Sanity check: one batch through the untrained model, to confirm the shapes

def check_shapes(model):
    x, y = get_batch("train")      # (B, T) token ids and their next-token targets
    logits = model.forward(x)      # (B, T, vocab_size)
    print(f"input {tuple(x.shape)} -> logits {tuple(logits.shape)}")


# -----------------------------------------------------------------------------
# Main: wires the steps together

def main():
    model = GPT()
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters")
    check_shapes(model)

    if MODE == "train":
        train(model)                 # learn the weights (and save them)
    elif MODE == "generate":
        load_trained_weights(model)  # reuse weights from an earlier training run
    else:
        raise ValueError(f"MODE must be 'train' or 'generate', not {MODE!r}")

    prompt = "SHAKESPEAR"  # text the model continues
    max_new_tokens = 100            # characters to generate
    print(complete(model, prompt, max_new_tokens))


if __name__ == "__main__":
    main()
