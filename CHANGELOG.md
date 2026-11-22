# Change log

Dated entries. Rules here are fixed before the data they govern. Nothing is edited silently.

## 2026-11-21, 23:40: registration (before any candidate file exists)

1. Concept and gates as in `PLAN.md` at this time. Byte limit 8,486,236,352 (unsloth UD-IQ2_XXS at `ae9327c`).
2. Release test (G4): the candidate must have lower KL than each of UD-IQ2_XXS, UD-IQ1_M and AD-IQ2_XXS-Q2 on T1 and on T2, with the 95% paired-bootstrap CI of the difference below zero, and a top-1 agreement point estimate not lower. AD-IQ2_M is over the limit and shown for context only. All claims name the reference: the sibling's Q8_0 conversion at P = 3.
3. G1 (offline): GPTQ-Q2_0 must beat the same encoder on the diagonal of the same Gram, on relative output error over the held-out calibration Gram, in layers 0 and 12. If it fails, the full capture of layers 13-23 is skipped and down projections use diag rounding.
4. G2 block: B3 (layers 12-17), fixed before any S score. Builds differ only in those six down tensors.
5. Reporting: rows per expert per layer, T2 per-chunk differences with leave-one-out, and G5 as a projection (no 16 GB Mac available).

## 2026-11-22, 01:10: G1 result (offline, held-out calibration Gram)

- Fit Gram: layers 0-12 from 240 chunks (123K tokens) of `data/calib-fit.txt`, forward on the sibling's cand.gguf. Eval Gram: 36 held-out conversations.
- Relative output error, diag control vs best GPTQ: layer 0 0.09879 vs 0.06931 (alpha 0.25); layer 12 0.11444 vs 0.08660 (alpha 0.25). `results/g1.tsv`. **G1: PASS.**
- alpha = 0.25 is fixed for every layer from here on. It was chosen on the calibration holdout, never on S, T1 or T2.
- Reference point: IQ4_NL down has about 0.004-0.005 on the pilot Grams, so Q2_0 stays far coarser even after GPTQ.
- Change of plan (from the sibling's measurements, Addendum H in its repo): at equal bytes the per-layer allocation matters more than rounding. First build M0 = UD-IQ2_XXS's per-layer types (gate_up, down Q2_0 vs 4.5-bpw, non-expert) with our own encodings, NVFP4 down replaced by IQ4_NL (same bytes, runs on Metal).

## 2026-11-22, 03:30: G2 result and what it changes

All on S (16 chunks, P = 3), paired bootstrap over chunks, from `results/kl.tsv`:

- **G2: FAIL.** B3 down (layers 12-17) as GPTQ-Q2_0 vs the matched diag control, everything else equal (M0 map): 0.10185 vs 0.09892, difference +0.00293 [+0.00200, +0.00387]. The offline gain (G1: 24-30% lower held-out output error) does not carry over to KL on S. GPTQ is dropped for the down projections; the registered rule stands.
- Decomposition at 8.49 GB (independent replication of the sibling's Addendum H): UD-IQ2_XXS 0.07064; M0 (UD's types, our encodings) 0.09791; M0 with UD's own gate_up tensors 0.08115. UD's gate_up encoding is worth 0.01676 [0.01507, 0.01837], the rest 0.01051 [0.00909, 0.01186].
- Importance source: our llama-quantize gate_up at M0's map with bartowski's imatrix instead of ggml-org's: +0.00009 [-0.00031, +0.00049]. No effect.
- X1 (UD-IQ2_XXS with our GPTQ-Q2_0 at its 17 Q2_0 downs and IQ4_NL at its 7 NVFP4 downs): 0.07715, +0.00651 [+0.00544, +0.00751] vs UD.
- Consequence: our own encoders lose to Unsloth's at every tensor class we tested. The remaining route to a better file at this size is allocation over Unsloth-encoded tensors (from several UD files) with an all-Metal constraint. Such a file is a derivative of Unsloth's work and must say so.

## 2026-11-22, 05:00: post hoc GPTQ v2 (registered before its data)

Decided after G2 failed and after these facts: X2 (UD-IQ2_XXS with only its 7 NVFP4 downs re-encoded as IQ4_NL, all-Metal) 0.07127 on S, +0.00063 [+0.00026, +0.00101] vs UD; X1 vs X2 +0.00588 [+0.00493, +0.00674]; bundle mix-early (UD Q2_0 at downs 18-19 funding gate_up 0, 1, 4 from UD-IQ3_XXS) +0.00515 [+0.00395, +0.00644] vs X2. Decoding the written donors reproduces G1 exactly (no packing bug). On the held-out conversations UD's own Q2_0 down has the error and bias of our diag rounding (results/bias-split.tsv), so GPTQ's offline gain does not reach KL. Leading explanation: domain shift.

1. Diagnostic only: the same encodings scored on a Gram from S text with the X2 forward (`scripts/sgram.sh`). Nothing is chosen on it.
2. GPTQ v2 (`scripts/gptq2.sh`): fit Grams on X2 itself; alpha from {0.25, 1, 4, 16} by the lowest mean output error over layers 0 and 12 on a Gram from WikiText-2 train (`data/calib-wt.txt`, the sibling's file, 0 shared 13-grams with S, T1, T2, T3), chosen by script; if no alpha beats diag there, stop.
3. X3 = X2 with GPTQ v2 at all 17 Q2_0 down slots. It becomes the candidate only if it beats X2 on S with the paired CI below zero. Then G4 on T1/T2 as registered. This step is post hoc and is reported as such.

## 2026-11-22, 05:10: amendment to the GPTQ v2 alpha rule (before any sweep result)

- alpha is chosen by the lowest mean of two held-out errors over layers 0 and 12: the WikiText-2 train Gram (`grams/wt-x2`) and the held-out conversations Gram (`grams/eval`), equal weight. A WikiText-only choice would align with S and T1.
- `scripts/gptq2.sh` (already running) picks on WikiText alone; `scripts/alpha-pool.sh` recomputes the pooled choice. If they differ, X3 is re-encoded with the pooled alpha before it is scored.
- G4 is unchanged: T1 and T2 must each pass with their own CI. T2 (reasoning traces) is the cross-domain guard and is reported separately.

## 2026-11-22, 06:00: GPTQ v2 result; mean-aware scale refit registered (before its data)

- GPTQ v2 (fit Grams on X2, pooled alpha = 1.0: mean held-out error 0.09865 vs 0.11299 for diag over WikiText train and conversations, layers 0 and 12): X3 = X2 with GPTQ-Q2_0 at its 17 Q2_0 downs scores 0.07513 on S, X3 - X2 +0.00386 [+0.00271, +0.00497]. **Fails**; X3 is not a candidate. GPTQ for the Q2_0 down projections is closed for this model.
- New lever, from the sibling's full-Gram finding (UD's gate_up beats llama-quantize's in 128/128 experts of layers 6 and 20, mostly through the expert-mean input direction): `scripts/meanfit.py` keeps llama-quantize's codes and fp16 d and refits only the 3-4 bit sub-block scales under n[diag(Var x) + m m^T] from the imatrix (closed form, then discrete coordinate descent; only scale bits change, decode checked).
- Test M0r: M0 with all 24 gate_up tensors refit, everything else equal. Reading: M0r vs M0 on S, paired; UD's gate_up (m0-udgu, 0.08115) is the reference for how much of the 0.01676 gap it closes. A mechanism check on the sibling's full gate_up Grams (layers 6 and 20, WikiText-2 train) runs first and decides nothing.
