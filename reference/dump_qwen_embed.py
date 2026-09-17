"""Dump a reference run of Qwen3-Embedding-0.6B, for EMBEDDING.md stage E1.

The oracle is transformers' own Qwen3Model in fp32 on CPU over the
checkpoint's model.safetensors. The architecture is the one
reference/dump_qwen.py already dumps for Z-Image's text encoder -- same
layer, same RoPE convention, same GQA -- so the layer-0 internals here are
named identically and the Go side walks them with the same test. What is new
is the *ends* of the model, which is where an embedding model differs from a
text encoder:

  * every layer runs, not NumLayers-1;
  * the final `norm` runs;
  * the pooled vector is the **last token's** row (1_Pooling asks for
    pooling_mode_lasttoken), and the last token is the `<|endoftext|>` the
    tokenizer's post-processor appends -- which this dump checks by id
    rather than assuming;
  * the vector is L2-normalised.

Three things this settles rather than assumes:

  * the tokenizer really does append 151643, and `add_special_tokens=False`
    really does suppress it, which is what says the append is the
    post-processor and not the prompt;
  * right padding does not change a real token's hidden state (the attention
    is causal), so a Go implementation that never pads is running the same
    model as a batched PyTorch one -- `pad_gap` is the measured difference;
  * the four texts on the model card produce the card's own similarity
    matrix, which is the end-to-end number every later stage is checked
    against.

    .venv/bin/python reference/dump_qwen_embed.py
"""

import argparse
import json
import os

import torch
from transformers import AutoModel, AutoTokenizer

# The model card's own example, which is the only end-to-end number about
# this checkpoint that was not produced by this script.
TASK = "Given a web search query, retrieve relevant passages that answer the query"
QUERIES = ["What is the capital of China?", "Explain gravity"]
DOCUMENTS = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives "
    "weight to physical objects and is responsible for the movement of planets "
    "around the sun.",
]
CARD_SCORES = [
    [0.7645568251609802, 0.14142508804798126],
    [0.13549736142158508, 0.5999549627304077],
]


def instruct(task, query):
    return f"Instruct: {task}\nQuery:{query}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-Embedding-0.6B")
    ap.add_argument("--out", default="reference/out/embed")
    ap.add_argument("--layer", type=int, default=0, help="layer whose internals are dumped")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModel.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cfg = model.config
    heads, kv_heads, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

    texts = [instruct(TASK, q) for q in QUERIES] + DOCUMENTS
    encoded = [tok(t).input_ids for t in texts]
    bare = tok(texts[0], add_special_tokens=False).input_ids

    manifest = {
        "texts": texts,
        "task": TASK,
        "queries": QUERIES,
        "documents": DOCUMENTS,
        "ids": encoded,
        "ids_no_special": bare,
        "eos_id": tok.eos_token_id,
        "appended_id": encoded[0][-1],
        "appends_eos": encoded[0] == bare + [encoded[0][-1]],
        "hidden_size": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": hd,
        "intermediate_size": cfg.intermediate_size,
        "rms_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_parameters["rope_theta"],
        "vocab_size": cfg.vocab_size,
        "dump_layer": args.layer,
        "card_scores": CARD_SCORES,
        "tensors": {},
    }
    print(f"appended token {manifest['appended_id']} "
          f"({tok.convert_ids_to_tokens([manifest['appended_id']])[0]!r}), "
          f"post-processor: {manifest['appends_eos']}")
    for t, ids in zip(texts, encoded):
        print(f"  {len(ids):3d} tokens  {t[:60]!r}")

    def dump(name, t):
        t = t.detach().contiguous().to(torch.float32)
        with open(os.path.join(args.out, name + ".bin"), "wb") as fh:
            fh.write(t.numpy().tobytes())
        flat = t.flatten()
        manifest["tensors"][name] = {
            "shape": list(t.shape),
            "count": int(flat.numel()),
            "sum": float(flat.double().sum()),
            "absmax": float(flat.abs().max()),
        }
        print(f"  {name:28s} {str(list(t.shape)):18s} "
              f"sum={flat.double().sum():+.5f} absmax={flat.abs().max():.4g}")

    # The walked text is the first one, unpadded and on its own: that is the
    # shape the Go side runs, one sequence with no mask.
    ids = torch.tensor([encoded[0]])
    T = ids.shape[1]
    manifest["seq"] = T
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states
    last = out.last_hidden_state
    manifest["hidden_states_len"] = len(hs)
    manifest["final_is_normed"] = float((hs[-1] - last).abs().max())

    # What last_hidden_state is: every layer, then `norm`. Recomputed by hand
    # so a Go port can be bisected rather than guessed at.
    with torch.no_grad():
        pos = torch.arange(T).unsqueeze(0)
        h = model.embed_tokens(ids)
        cos, sin = model.rotary_emb(h, pos)
        mask = torch.full((T, T), float("-inf")).triu(1)[None, None]
        for i in range(cfg.num_hidden_layers):
            h = model.layers[i](h, attention_mask=mask, position_embeddings=(cos, sin))
        hand_layers = h
        hand = model.norm(h)
    manifest["hand_gap"] = float((hand - last).abs().max())
    # transformers v5 captures hidden states *after* the final norm, so
    # hs[-1] is last_hidden_state and not layer 27's output. The two differ
    # by `prenorm_gap` here, which is 775 -- a port validated against hs[-1]
    # as "the last layer" would be checking the wrong tensor. `prenorm`
    # below is therefore the recomputed one.
    manifest["prenorm_gap"] = float((hand_layers - hs[-1]).abs().max())
    print(f"last_hidden_state vs {cfg.num_hidden_layers} layers + norm by hand: "
          f"{manifest['hand_gap']:.3g}")

    # Padding: batched PyTorch pads and masks, the Go side does neither.
    # Right padding cannot reach a real token through a causal mask, and this
    # is the measurement that says so.
    with torch.no_grad():
        padded = tok(texts, padding=True, padding_side="right", return_tensors="pt")
        pm = padded.attention_mask.bool()
        po = model(input_ids=padded.input_ids, attention_mask=pm).last_hidden_state
        padded_first = po[0][pm[0]]
    manifest["pad_gap"] = float((padded_first - last[0]).abs().max())
    manifest["pad_rms"] = float(last[0].pow(2).mean().sqrt())
    print(f"right-padded batch vs unpadded: max abs {manifest['pad_gap']:.3g} "
          f"against rms {manifest['pad_rms']:.4g}")

    # Layer internals, recomputed from the same weights. Identical to
    # dump_qwen.py's, tensor for tensor, so both walk the same test.
    layer = model.layers[args.layer]
    a = layer.self_attn
    with torch.no_grad():
        x = hs[args.layer]
        normed = layer.input_layernorm(x)
        q = a.q_proj(normed).view(1, T, heads, hd)
        k = a.k_proj(normed).view(1, T, kv_heads, hd)
        v = a.v_proj(normed).view(1, T, kv_heads, hd)
        qn, kn = a.q_norm(q), a.k_norm(k)

        def rope(t):  # NeoX style: the halves rotate against each other
            c, s = cos.unsqueeze(2), sin.unsqueeze(2)
            half = t.shape[-1] // 2
            rot = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
            return t * c + rot * s

        qr, kr = rope(qn), rope(kn)
        rep = heads // kv_heads
        kx = kr.repeat_interleave(rep, dim=2)
        vx = v.repeat_interleave(rep, dim=2)
        scores = torch.einsum("bthd,bshd->bhts", qr, kx) * (hd ** -0.5)
        scores = scores + torch.full((T, T), float("-inf")).triu(1)[None, None]
        probs = scores.softmax(-1, dtype=torch.float32)
        ctx = torch.einsum("bhts,bshd->bthd", probs, vx).reshape(1, T, heads * hd)
        attn_out = a.o_proj(ctx)
        resid1 = x + attn_out
        normed2 = layer.post_attention_layernorm(resid1)
        gate = torch.nn.functional.silu(layer.mlp.gate_proj(normed2))
        swiglu = gate * layer.mlp.up_proj(normed2)
        mlp = layer.mlp.down_proj(swiglu)
        layer_out = resid1 + mlp

    print("tensors:")
    dump("input_ids", ids.to(torch.float32))
    dump("cos", cos[0])
    dump("sin", sin[0])
    dump("embeddings", hs[0][0])
    dump("input_layernorm", normed[0])
    dump("q", q.reshape(T, heads * hd))
    dump("k", k.reshape(T, kv_heads * hd))
    dump("v", v.reshape(T, kv_heads * hd))
    dump("q_normed", qn.reshape(T, heads * hd))
    dump("k_normed", kn.reshape(T, kv_heads * hd))
    dump("q_roped", qr.reshape(T, heads * hd))
    dump("k_roped", kr.reshape(T, kv_heads * hd))
    dump("attn_ctx", ctx[0])
    dump("attn_out", attn_out[0])
    dump("resid1", resid1[0])
    dump("post_attention_layernorm", normed2[0])
    dump("swiglu", swiglu[0])
    dump("mlp", mlp[0])
    dump("layer_out", layer_out[0])
    dump("hidden_1", hs[1][0])
    dump("hidden_2", hs[2][0])
    dump("hidden_14", hs[14][0])
    dump("prenorm", hand_layers[0])
    dump("final", last[0])

    # The ends of the model: pooling and normalisation, on the walked text.
    pooled = last[0, -1]
    unit = torch.nn.functional.normalize(pooled, p=2, dim=0)
    dump("pooled", pooled)
    dump("embedding", unit)

    # And all four texts, one at a time and unpadded -- which is the run the
    # Go side makes -- so the similarity matrix is reproducible without a
    # batch.
    vectors = []
    with torch.no_grad():
        for t_ids in encoded:
            h = model(input_ids=torch.tensor([t_ids])).last_hidden_state[0, -1]
            vectors.append(torch.nn.functional.normalize(h, p=2, dim=0))
    vecs = torch.stack(vectors)
    dump("embeddings_all", vecs)
    scores = (vecs[:2] @ vecs[2:].T)
    manifest["scores"] = scores.tolist()
    manifest["card_gap"] = float((scores - torch.tensor(CARD_SCORES)).abs().max())
    print(f"similarity matrix {scores.tolist()}")
    print(f"  against the model card: {manifest['card_gap']:.3g}")

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest['tensors'])} tensors to {args.out}")


if __name__ == "__main__":
    main()
