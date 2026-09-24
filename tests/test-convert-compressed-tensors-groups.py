#!/usr/bin/env python3
"""
Converter test: a compressed-tensors checkpoint with several config groups.

Builds a tiny Qwen2-MoE checkpoint quantized like Qwen3.8-Flash-Next MXFP4/FP8 exports:
  group_0  "Linear"                              MXFP4 (4-bit float, group 32, packed)
  group_1  regex on mlp.shared_expert.down_proj  FP8 e4m3, block 128x128, `weight_scale`
  group_2  regex on self_attn q/k/v/o            FP8 e4m3, block 128x128, `weight_scale_inv`,
           anchored on the checkpoint name (model.language_model.), which the converter strips
  ignore   lm_head, embeddings, router, shared expert gate, norms (stored as BF16)

and checks the converted GGUF:
  - every group_0 tensor is MXFP4 and bit-exact: the raw block bytes equal an independent
    repack, and the decoded values equal an independent decode of the source codes
  - group_1 beats the "Linear" class for shared_expert.down_proj (regex over class)
  - FP8 tensors are F16 and equal the block-scaled dequant rounded to F16
    (Q8_0 with --fp8-as-q8, within Q8_0 error)
  - ignored tensors pass through unchanged
  - the file is labeled MXFP4_MOE

usage: python tests/test-convert-compressed-tensors-groups.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gguf-py"))

import gguf  # noqa: E402
from conversion import load_all_models  # noqa: E402
from conversion.base import ModelBase  # noqa: E402

HIDDEN = 192       # not a multiple of 128: the FP8 blocks are partial
N_HEAD = 3
N_HEAD_KV = 1
HEAD_DIM = 64
N_FF_EXP = 96
N_FF_SHEXP = 96
N_EXPERT = 4
N_LAYER = 2
N_VOCAB = 256
BLOCK = [128, 128]

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)

rng = np.random.default_rng(1234)


def make_mxfp4(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """compressed-tensors MXFP4: packed [rows, cols/2] (element 2i low nibble), scale [rows, cols/32] E8M0.
    Returns (packed, scale, reference ggml block bytes, reference values)."""
    codes = rng.integers(0, 16, size=(rows, cols), dtype=np.uint8)
    exps = rng.integers(100, 150, size=(rows, cols // 32), dtype=np.uint8)
    exps[0, 0], exps[-1, -1] = 0, 1  # the E8M0 subnormal-half corner cases
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)

    # ggml block_mxfp4: e, then qs[j] = code[j] | code[j + 16] << 4
    blk = codes.reshape(rows, cols // 32, 32)
    qs = blk[:, :, :16] | (blk[:, :, 16:] << 4)
    ref_bytes = np.concatenate([exps[:, :, None], qs], axis=-1).reshape(rows, (cols // 32) * 17)

    ref_vals = E2M1[codes] * np.repeat(np.ldexp(np.float64(1), exps.astype(np.int64) - 127), 32, axis=1)
    return torch.from_numpy(packed), torch.from_numpy(exps), ref_bytes, ref_vals.astype(np.float32)


def make_fp8(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """FP8 e4m3 weight with a BF16 block scale; returns (weight, scale, reference f32 values)."""
    w = torch.from_numpy(rng.uniform(-448, 448, size=(rows, cols)).astype(np.float32)).to(torch.float8_e4m3fn)
    sr, sc = -(-rows // BLOCK[0]), -(-cols // BLOCK[1])
    s = torch.from_numpy(rng.uniform(1e-4, 1e-2, size=(sr, sc)).astype(np.float32)).to(torch.bfloat16)
    full = s.float().repeat_interleave(BLOCK[0], 0).repeat_interleave(BLOCK[1], 1)[:rows, :cols]
    return w, s, (w.float() * full).numpy()


def bf16(*shape: int) -> torch.Tensor:
    return torch.from_numpy(rng.normal(0, 0.02, size=shape).astype(np.float32)).to(torch.bfloat16)


def build_checkpoint(d: Path) -> dict[str, tuple[str, np.ndarray]]:
    """Writes the checkpoint; returns gguf name -> (expected kind, expected values or bytes)."""
    t: dict[str, torch.Tensor] = {}
    expect: dict[str, tuple[str, np.ndarray]] = {}
    pre = "model.language_model"

    t["lm_head.weight"] = bf16(N_VOCAB, HIDDEN)
    t[f"{pre}.embed_tokens.weight"] = bf16(N_VOCAB, HIDDEN)
    t[f"{pre}.norm.weight"] = bf16(HIDDEN)
    expect["output.weight"] = ("plain", t["lm_head.weight"].float().numpy())
    expect["token_embd.weight"] = ("plain", t[f"{pre}.embed_tokens.weight"].float().numpy())

    for il in range(N_LAYER):
        lp = f"{pre}.layers.{il}"
        t[f"{lp}.input_layernorm.weight"] = bf16(HIDDEN)
        t[f"{lp}.post_attention_layernorm.weight"] = bf16(HIDDEN)

        for proj, rows, cols, gg in (
            ("q_proj", N_HEAD * HEAD_DIM, HIDDEN, "attn_q"),
            ("k_proj", N_HEAD_KV * HEAD_DIM, HIDDEN, "attn_k"),
            ("v_proj", N_HEAD_KV * HEAD_DIM, HIDDEN, "attn_v"),
            ("o_proj", HIDDEN, N_HEAD * HEAD_DIM, "attn_output"),
        ):
            w, s, ref = make_fp8(rows, cols)
            t[f"{lp}.self_attn.{proj}.weight"] = w
            t[f"{lp}.self_attn.{proj}.weight_scale_inv"] = s
            expect[f"blk.{il}.{gg}.weight"] = ("fp8", ref)

        t[f"{lp}.mlp.gate.weight"] = bf16(N_EXPERT, HIDDEN)
        t[f"{lp}.mlp.shared_expert_gate.weight"] = bf16(1, HIDDEN)
        expect[f"blk.{il}.ffn_gate_inp.weight"] = ("plain", t[f"{lp}.mlp.gate.weight"].float().numpy())

        for proj, rows, cols, gg in (
            ("gate_proj", N_FF_SHEXP, HIDDEN, "ffn_gate_shexp"),
            ("up_proj", N_FF_SHEXP, HIDDEN, "ffn_up_shexp"),
        ):
            p, s, rb, rv = make_mxfp4(rows, cols)
            t[f"{lp}.mlp.shared_expert.{proj}.weight_packed"] = p
            t[f"{lp}.mlp.shared_expert.{proj}.weight_scale"] = s
            expect[f"blk.{il}.{gg}.weight"] = ("mxfp4", (rb, rv))  # type: ignore

        w, s, ref = make_fp8(HIDDEN, N_FF_SHEXP)
        t[f"{lp}.mlp.shared_expert.down_proj.weight"] = w
        t[f"{lp}.mlp.shared_expert.down_proj.weight_scale"] = s
        expect[f"blk.{il}.ffn_down_shexp.weight"] = ("fp8", ref)

        for proj, rows, cols, gg in (
            ("gate_proj", N_FF_EXP, HIDDEN, "ffn_gate_exps"),
            ("up_proj", N_FF_EXP, HIDDEN, "ffn_up_exps"),
            ("down_proj", HIDDEN, N_FF_EXP, "ffn_down_exps"),
        ):
            rbs, rvs = [], []
            for xid in range(N_EXPERT):
                p, s, rb, rv = make_mxfp4(rows, cols)
                t[f"{lp}.mlp.experts.{xid}.{proj}.weight_packed"] = p
                t[f"{lp}.mlp.experts.{xid}.{proj}.weight_scale"] = s
                rbs.append(rb)
                rvs.append(rv)
            expect[f"blk.{il}.{gg}.weight"] = ("mxfp4", (np.stack(rbs), np.stack(rvs)))  # type: ignore

    fp8_block = {
        "num_bits": 8, "type": "float", "symmetric": True, "strategy": "block",
        "block_structure": BLOCK, "dynamic": False,
    }
    config = {
        "architectures": ["Qwen2MoeForCausalLM"],
        "model_type": "qwen2_moe",
        "hidden_size": HIDDEN,
        "intermediate_size": N_FF_SHEXP,
        "moe_intermediate_size": N_FF_EXP,
        "shared_expert_intermediate_size": N_FF_SHEXP,
        "num_attention_heads": N_HEAD,
        "num_key_value_heads": N_HEAD_KV,
        "num_hidden_layers": N_LAYER,
        "num_experts": N_EXPERT,
        "num_experts_per_tok": 2,
        "vocab_size": N_VOCAB,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "quantization_status": "compressed",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4, "type": "float", "symmetric": True, "strategy": "group",
                        "group_size": 32, "dynamic": False, "block_structure": None,
                    },
                },
                "group_1": {
                    "targets": ["re:.*\\.mlp\\.shared_expert\\.down_proj$"],
                    "weights": fp8_block,
                },
                "group_2": {
                    "targets": ["re:model\\.language_model\\.layers\\.\\d+\\.self_attn\\.(q_proj|k_proj|v_proj|o_proj)$"],
                    "weights": fp8_block,
                },
            },
            "ignore": [
                "lm_head",
                "re:.*embed_tokens.*",
                "re:.*\\.mlp\\.gate$",
                "re:.*shared_expert_gate$",
                "re:.*norm.*",
            ],
        },
    }
    (d / "config.json").write_text(json.dumps(config))
    save_file(t, str(d / "model.safetensors"))
    return expect


def convert(d: Path, out: Path, fp8_as_q8: bool) -> gguf.GGUFReader:
    load_all_models()
    cls = ModelBase.from_model_architecture("Qwen2MoeForCausalLM")

    class NoVocab(cls):  # the test checkpoint has no tokenizer
        model_arch = cls.model_arch

        def set_vocab(self):
            pass

    model = NoVocab(d, gguf.LlamaFileType.MOSTLY_BF16, out, fp8_as_q8=fp8_as_q8)
    model.write()
    return gguf.GGUFReader(out)


def check(reader: gguf.GGUFReader, expect: dict[str, tuple[str, object]], fp8_as_q8: bool) -> list[str]:
    fails: list[str] = []
    tensors = {t.name: t for t in reader.tensors}
    ftype = int(reader.get_field(gguf.Keys.General.FILE_TYPE).parts[-1][0])
    if ftype != gguf.LlamaFileType.MOSTLY_MXFP4_MOE:
        fails.append(f"file type {ftype}, expected MOSTLY_MXFP4_MOE")

    n_mxfp4 = n_fp8 = 0
    for name, (kind, ref) in expect.items():
        if name not in tensors:
            fails.append(f"{name}: missing")
            continue
        t = tensors[name]
        qtype = gguf.GGMLQuantizationType(t.tensor_type)
        raw = np.asarray(t.data)
        if kind == "mxfp4":
            ref_bytes, ref_vals = ref  # type: ignore
            if qtype != gguf.GGMLQuantizationType.MXFP4:
                fails.append(f"{name}: {qtype.name}, expected MXFP4")
                continue
            if raw.shape != ref_bytes.shape or not np.array_equal(raw, ref_bytes):
                fails.append(f"{name}: block bytes differ from the reference repack")
            vals = gguf.quants.dequantize(raw, qtype)
            if not np.array_equal(vals, ref_vals):
                fails.append(f"{name}: decoded values differ, max |diff| {np.abs(vals - ref_vals).max()}")
            n_mxfp4 += 1
        elif kind == "fp8":
            want = gguf.GGMLQuantizationType.Q8_0 if fp8_as_q8 else gguf.GGMLQuantizationType.F16
            if qtype != want:
                fails.append(f"{name}: {qtype.name}, expected {want.name}")
                continue
            vals = gguf.quants.dequantize(raw, qtype).reshape(ref.shape)  # type: ignore
            if fp8_as_q8:
                # Q8_0 is within half a step of its per-32 absmax / 127 scale
                blocks = np.abs(ref).reshape(ref.shape[0], -1, 32).max(-1, keepdims=True) / 127  # type: ignore
                tol = np.repeat(blocks, 32, axis=-1).reshape(ref.shape) * 0.5 + 1e-3 * np.abs(ref)  # type: ignore
                if np.any(np.abs(vals - ref) > tol):
                    fails.append(f"{name}: Q8_0 error above half a step")
            elif not np.array_equal(vals, np.asarray(ref, dtype=np.float16).astype(np.float32)):
                fails.append(f"{name}: F16 differs from the dequantized FP8, max |diff| {np.abs(vals - ref).max()}")
            n_fp8 += 1
        else:
            vals = gguf.quants.dequantize(raw, qtype).reshape(ref.shape)  # type: ignore
            if not np.array_equal(vals, ref):
                fails.append(f"{name}: {qtype.name} differs from the BF16 source")

    print(f"  checked {n_mxfp4} MXFP4, {n_fp8} FP8, {len(expect) - n_mxfp4 - n_fp8} plain tensors")  # noqa: NP100
    return fails


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "ckpt"
        d.mkdir()
        expect = build_checkpoint(d)
        for fp8_as_q8 in (False, True):
            label = "--fp8-as-q8" if fp8_as_q8 else "default"
            reader = convert(d, Path(tmp) / f"out-{label}.gguf", fp8_as_q8)
            fails = check(reader, expect, fp8_as_q8)
            print(f"[{'PASS' if not fails else 'FAIL'}] {label}")  # noqa: NP100
            for f in fails:
                print(f"    {f}")  # noqa: NP100
            ok = ok and not fails
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
