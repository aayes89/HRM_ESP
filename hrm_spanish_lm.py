#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM-style Spanish Text LM (single-file) made by (Slam 2025)
- Tokenizador byte-level UTF-8 (0..255) + especiales.
- Planner (alto nivel): Encoder pequeño -> embedding de plan (Gumbel-softmax opcional) condicionado por prompt.
- Executor (bajo nivel): Decoder Transformer causal con prefijo [PLAN] (plan embedding proyectado a token).
- Preentrenamiento: LM puro sobre corpus (sin prompt explícito, se usa el propio segmento).
- Fine-tuning: con prompts (archivo TSV: prompt \t target) o simple LM si falta target.
- Generación: top-k/top-p, temperatura, longitud máx.

Requisitos: Python 3.9+, PyTorch 2.x con CUDA opcional.
"""
import math, os, argparse, random
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Tokenizer byte-level
# ---------------------------
class ByteTokenizer:
    def __init__(self):
        # 0..255 bytes + especiales
        self.PAD = 256
        self.BOS = 257
        self.EOS = 258
        self.PLAN = 259  # token sintético para inyectar el plan
        self.vocab_size = 260

    def encode(self, s: str, add_specials=False) -> List[int]:
        b = s.encode('utf-8', errors='replace')
        ids = list(b)
        if add_specials:
            return [self.BOS] + ids + [self.EOS]
        return ids

    def decode(self, ids: List[int]) -> str:
        bytes_out = bytearray([i for i in ids if 0 <= i <= 255])
        return bytes_out.decode('utf-8', errors='replace')

# ---------------------------
# Data loaders
# ---------------------------
def load_corpus_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return [line.rstrip('\n') for line in f]

def dataset_lm(tokenizer: ByteTokenizer, path: str, add_specials=True) -> List[List[int]]:
    lines = load_corpus_lines(path)
    # Concatenado ingenuo por línea -> secuencia por línea
    return [tokenizer.encode(line, add_specials=add_specials) for line in lines if line.strip()]

def dataset_tsv(tokenizer: ByteTokenizer, path: str) -> List[Tuple[List[int], List[int]]]:
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip(): 
                continue
            if '\t' not in line:
                # si no hay tab, se trata como LM puro
                x = tokenizer.encode(line.strip(), add_specials=True)
                rows.append((x, []))
            else:
                p, y = line.rstrip('\n').split('\t', 1)
                p_ids = tokenizer.encode(p, add_specials=True)
                y_ids = tokenizer.encode(y, add_specials=True)
                rows.append((p_ids, y_ids))
    return rows

# ---------------------------
# Transformer blocks
# ---------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.g * x * norm

class MHA(nn.Module):
    def __init__(self, d, n_heads, causal=True, cross=False):
        super().__init__()
        assert d % n_heads == 0, "d_model debe ser múltiplo de n_heads"
        self.d = d
        self.nh = n_heads
        self.hd = d // n_heads
        self.causal = causal
        self.cross = cross
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
    def forward(self, x, kv=None):
        # x: [B,T,D]; kv: [B,S,D] for cross
        q = self.q(x)
        if self.cross and kv is not None:
            k = self.k(kv); v = self.v(kv)
        else:
            k = self.k(x); v = self.v(x)
        B, T, D = x.size(); S = k.size(1)
        q = q.view(B,T,self.nh,self.hd).transpose(1,2) # [B,nh,T,hd]
        k = k.view(B,S,self.nh,self.hd).transpose(1,2) # [B,nh,S,hd]
        v = v.view(B,S,self.nh,self.hd).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.hd) # [B,nh,T,S]
        if self.causal and kv is None:
            # máscara causal T x S con mismo dtype que att
            mask = torch.triu(torch.full((T, S), torch.finfo(att.dtype).min, device=x.device, dtype=att.dtype), diagonal=1)
            att = att + mask
        w = att.softmax(dim=-1)
        z = w @ v # [B,nh,T,hd]
        z = z.transpose(1,2).contiguous().view(B,T,D)
        return self.o(z)

class FFN(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        self.up = nn.Linear(d, d*mult*2, bias=False)
        self.down = nn.Linear(d*mult, d, bias=False)
    def forward(self, x):
        u, v = self.up(x).chunk(2, dim=-1)  # GEGLU
        return self.down(F.gelu(u) * v)

class DecoderBlock(nn.Module):
    def __init__(self, d, n_heads, ff_mult=4, cross_mem=False):
        super().__init__()
        self.n1 = RMSNorm(d); self.att = MHA(d, n_heads, causal=True, cross=False)
        self.n2 = RMSNorm(d)
        self.cross_mem = cross_mem
        if cross_mem:
            self.cross = MHA(d, n_heads, causal=False, cross=True)
            self.nx = RMSNorm(d)
        self.ff = FFN(d, ff_mult)
    def forward(self, x, mem=None):
        x = x + self.att(self.n1(x))
        if self.cross_mem and mem is not None:
            x = x + self.cross(self.nx(x), kv=mem)
        x = x + self.ff(self.n2(x))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, d, n_heads, ff_mult=4):
        super().__init__()
        self.n1 = RMSNorm(d); self.att = MHA(d, n_heads, causal=False, cross=False)
        self.n2 = RMSNorm(d); self.ff = FFN(d, ff_mult)
    def forward(self, x):
        x = x + self.att(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x

# ---------------------------
# HRM-like model
# ---------------------------
@dataclass
class HRMConfig:
    vocab_size: int = 260
    d_model: int = 512
    n_heads: int = 8
    n_layers_enc: int = 2
    n_layers_dec: int = 8
    ff_mult: int = 4
    max_len: int = 1024
    plan_dim: int = 256
    n_plan_codes: int = 32  # opcional: discretización suave
    use_gumbel: bool = True
    p_dropout: float = 0.1

class Planner(nn.Module):
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([EncoderBlock(cfg.d_model, cfg.n_heads, cfg.ff_mult) for _ in range(cfg.n_layers_enc)])
        self.norm = RMSNorm(cfg.d_model)
        self.proj_plan = nn.Linear(cfg.d_model, cfg.plan_dim)
        self.codebook = nn.Parameter(torch.randn(cfg.n_plan_codes, cfg.plan_dim) * 0.02)
        self.use_gumbel = cfg.use_gumbel
        self.dropout = nn.Dropout(cfg.p_dropout)
    def forward(self, x):
        # x: [B,T] prompt_ids
        B, T = x.size()
        T = min(T, self.pos.num_embeddings)
        pos = torch.arange(T, device=x.device)
        h = self.tok(x[:, :T]) + self.pos(pos)[None, :, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        # resumen: mean pooling + proy
        h_mean = h.mean(dim=1)
        plan = self.proj_plan(self.dropout(h_mean))  # [B,plan_dim]
        # discretización opcional (Gumbel-softmax sobre codebook)
        if self.use_gumbel:
            logits = F.linear(F.normalize(plan, dim=-1), F.normalize(self.codebook, dim=-1))  # [B,K]
            g = F.gumbel_softmax(logits, tau=1.0, hard=False)  # [B,K]
            plan_vec = g @ self.codebook  # mezcla suave
            return plan_vec, logits
        return plan, None

class Executor(nn.Module):
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len+1, cfg.d_model)  # +1 por [PLAN]
        self.plan_to_token = nn.Linear(cfg.plan_dim, cfg.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(cfg.d_model, cfg.n_heads, cfg.ff_mult, cross_mem=False) for _ in range(cfg.n_layers_dec)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.dropout = nn.Dropout(cfg.p_dropout)
    def forward(self, y, plan_vec):
        # y: [B,T] (incluye BOS..)
        # plan_vec: [B,plan_dim] -> se inyecta como primer "token" (prefijo)
        B, T = y.size()
        plan_tok = self.plan_to_token(plan_vec).unsqueeze(1)  # [B,1,D]
        Tpos = min(T+1, self.pos.num_embeddings)
        pos = torch.arange(Tpos, device=y.device)
        y_emb = self.tok(y[:, :Tpos-1])
        h = torch.cat([plan_tok, y_emb], dim=1) + self.pos(pos)[None,:,:]
        h = self.dropout(h)
        for blk in self.blocks:
            h = blk(h, mem=None)
        h = self.norm(h)
        # Descartar la posición del [PLAN] al predecir
        logits = self.lm_head(h[:, 1:, :])  # [B,T,V]
        return logits

class HRMModel(nn.Module):
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.cfg = cfg
        self.planner = Planner(cfg)
        self.executor = Executor(cfg)
    def forward(self, prompt_ids, y_in):
        # prompt_ids: [B,TP], y_in: [B,TY] -> logits [B,TY,V]
        plan_vec, logits_codes = self.planner(prompt_ids)
        logits = self.executor(y_in, plan_vec)
        return logits, logits_codes

# ---------------------------
# Training utils
# ---------------------------
def batchify(seq_list: List[List[int]], B: int, max_len: int, pad: int, device):
    # corta/acolcha a max_len
    random.shuffle(seq_list)
    for i in range(0, len(seq_list), B):
        batch = seq_list[i:i+B]
        T = min(max_len, max(len(s) for s in batch))
        x = torch.full((len(batch), T), pad, dtype=torch.long, device=device)
        for j, s in enumerate(batch):
            s = s[:T]
            x[j, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        yield x

def tsv_batchify(pairs: List[Tuple[List[int], List[int]]], B: int, max_p: int, max_y: int, pad: int, device):
    random.shuffle(pairs)
    for i in range(0, len(pairs), B):
        chunk = pairs[i:i+B]
        Tp = min(max_p, max(len(p) for p,_ in chunk))
        Ty = min(max_y, max((len(y) if len(y)>0 else len(p)) for p,y in chunk))
        P = torch.full((len(chunk), Tp), pad, dtype=torch.long, device=device)
        Y = torch.full((len(chunk), Ty), pad, dtype=torch.long, device=device)
        for j, (p, y) in enumerate(chunk):
            pp = p[:Tp]
            yy = (y if len(y)>0 else p)[:Ty]
            P[j, :len(pp)] = torch.tensor(pp, device=device)
            Y[j, :len(yy)] = torch.tensor(yy, device=device)
        yield P, Y

def lm_loss(logits, targets, pad_id):
    # logits: [B,T,V], targets: [B,T]
    # shift: predecir t+1
    logits = logits[:, :-1, :].contiguous()
    targets = targets[:, 1:].contiguous()
    V = logits.size(-1)
    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=pad_id)
    return loss

# ---------------------------
# Generation
# ---------------------------
@torch.no_grad()
def generate(model: HRMModel, tokenizer: ByteTokenizer, prompt: str, max_new_tokens=200, temperature=0.9, top_k=50, top_p=0.95, device='cpu'):
    model.eval()
    p_ids = torch.tensor([tokenizer.encode(prompt, add_specials=True)], device=device)
    plan_vec, _ = model.planner(p_ids)
    y = torch.tensor([[tokenizer.BOS]], device=device)
    for _ in range(max_new_tokens):
        logits = model.executor(y, plan_vec)  # [1,T,V]
        next_logits = logits[:, -1, :] / max(1e-6, temperature)
        # top-k
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = torch.finfo(next_logits.dtype).min
        # top-p
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask = cum > top_p
            mask[..., 0] = False
            sorted_logits[mask] = torch.finfo(sorted_logits.dtype).min
            next_logits = torch.zeros_like(next_logits).scatter_(1, sorted_idx, sorted_logits)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # [1,1]
        y = torch.cat([y, next_id], dim=1)
        if next_id.item() == tokenizer.EOS:
            break
    return tokenizer.decode(y[0,1:].tolist())

# ---------------------------
# Train loops
# ---------------------------
def train_lm(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    tk = ByteTokenizer()

    # Dataset minimalista: sliding window por archivo completo
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, file_path, tokenizer, max_len):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # Para estabilidad autoregresiva, incluimos BOS al inicio del buffer
            self.tokens = [tokenizer.BOS] + tokenizer.encode(text, add_specials=False) + [tokenizer.EOS]
            self.max_len = max_len
            self.tokenizer = tokenizer
        def __len__(self):
            return max(0, len(self.tokens) - self.max_len - 1)
        def __getitem__(self, idx):
            x = self.tokens[idx:idx + self.max_len]
            y = self.tokens[idx + 1:idx + self.max_len + 1]
            return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    dataset = TextDataset(args.corpus, tk, args.max_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    cfg = HRMConfig(
        vocab_size=tk.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers_enc=args.n_layers_enc,
        n_layers_dec=args.n_layers_dec,
        ff_mult=4,
        max_len=args.max_len,
        plan_dim=args.plan_dim,
        n_plan_codes=args.n_plan_codes,
        use_gumbel=(not args.no_gumbel),
        p_dropout=args.dropout,
    )
    model = HRMModel(cfg).to(device)

    os.makedirs(args.out_dir, exist_ok=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.98), weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(device=='cuda'))

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            # Para preentrenamiento: el planner ve el mismo segmento (condición débil)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device=='cuda')):
                logits, logits_codes = model(x, x)  # prompt_ids = x, y_in = x
                loss = lm_loss(logits, y, pad_id=tk.PAD)
                if logits_codes is not None:
                    code_loss = 0.001 * (-(F.log_softmax(logits_codes, dim=-1)).mean())
                    loss = loss + code_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"[pretrain {epoch+1}/{args.epochs}] step {global_step} | loss {loss.item():.4f}")

        ckpt_path = os.path.join(args.out_dir, f"pretrain_epoch{epoch+1}.pt")
        torch.save({'cfg': cfg.__dict__, 'model': model.state_dict()}, ckpt_path)

    # último
    torch.save({'cfg': cfg.__dict__, 'model': model.state_dict()}, os.path.join(args.out_dir, "pretrain_last.pt"))
    print("Entrenamiento finalizado.")

def train_ft(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    tk = ByteTokenizer()
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = HRMConfig(**ckpt['cfg'])
    model = HRMModel(cfg).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    pairs = dataset_tsv(tk, args.tsv)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.98), weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))
    model.train()
    os.makedirs(args.out_dir, exist_ok=True)
    global_step = 0
    for epoch in range(args.epochs):
        for P, Y in tsv_batchify(pairs, args.batch_size, args.max_len, args.max_len, tk.PAD, device):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device=='cuda')):
                logits, logits_codes = model(P, Y if (Y is not None and Y.numel()>0) else P)
                loss = lm_loss(logits, Y if (Y is not None and Y.numel()>0) else P, tk.PAD)
                if logits_codes is not None:
                    code_loss = 0.001 * (-(F.log_softmax(logits_codes, dim=-1)).mean())
                    loss = loss + code_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"[finetune {epoch+1}/{args.epochs}] step {global_step} | loss {loss.item():.4f}")
        torch.save({'cfg': cfg.__dict__, 'model': model.state_dict()}, os.path.join(args.out_dir, f"ft_epoch{epoch+1}.pt"))
    torch.save({'cfg': cfg.__dict__, 'model': model.state_dict()}, os.path.join(args.out_dir, "ft_last.pt"))

# ---------------------------
# CLI
# ---------------------------
def cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    # pretrain
    sp = sub.add_parser("pretrain")
    sp.add_argument("--corpus", required=True, help="TXT con español (puede ser multilinea)")
    sp.add_argument("--out_dir", default="checkpoints")
    sp.add_argument("--epochs", type=int, default=1)
    sp.add_argument("--batch_size", type=int, default=16)
    sp.add_argument("--max_len", type=int, default=512)
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--d_model", type=int, default=512)
    sp.add_argument("--n_heads", type=int, default=8)
    sp.add_argument("--n_layers_enc", type=int, default=2)
    sp.add_argument("--n_layers_dec", type=int, default=8)
    sp.add_argument("--plan_dim", type=int, default=256)
    sp.add_argument("--n_plan_codes", type=int, default=32)
    sp.add_argument("--no_gumbel", action="store_true")
    sp.add_argument("--dropout", type=float, default=0.1)
    sp.add_argument("--log_every", type=int, default=50)
    sp.add_argument("--cpu", action="store_true")

    # finetune
    sf = sub.add_parser("finetune")
    sf.add_argument("--checkpoint", required=True)
    sf.add_argument("--tsv", required=True, help="prompt\\ttarget por línea (UTF-8)")
    sf.add_argument("--out_dir", default="checkpoints_ft")
    sf.add_argument("--epochs", type=int, default=1)
    sf.add_argument("--batch_size", type=int, default=8)
    sf.add_argument("--max_len", type=int, default=512)
    sf.add_argument("--lr", type=float, default=1e-4)
    sf.add_argument("--log_every", type=int, default=50)
    sf.add_argument("--cpu", action="store_true")

    # generate
    sg = sub.add_parser("generate")
    sg.add_argument("--checkpoint", required=True)
    sg.add_argument("--prompt", required=True)
    sg.add_argument("--max_new_tokens", type=int, default=200)
    sg.add_argument("--temperature", type=float, default=0.9)
    sg.add_argument("--top_k", type=int, default=50)
    sg.add_argument("--top_p", type=float, default=0.95)
    sg.add_argument("--cpu", action="store_true")

    args = p.parse_args()
    if args.cmd == "pretrain":
        train_lm(args)
    elif args.cmd == "finetune":
        train_ft(args)
    elif args.cmd == "generate":
        device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        tk = ByteTokenizer()
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg = HRMConfig(**ckpt['cfg'])
        model = HRMModel(cfg).to(device)
        model.load_state_dict(ckpt['model'], strict=True)
        out = generate(model, tk, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p, device)
        print(out)
    else:
        print("Usa: pretrain | finetune | generate")

if __name__ == "__main__":
    cli()
