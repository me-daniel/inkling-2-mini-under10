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

## 2026-11-22, 06:40: reading rules for M0r (before its score; from the Fable direction review)

- M0r is a mechanism test. Even UD's own gate_up leaves M0 at 0.08115, 0.01051 above UD, so no M0-family file can be released from it.
- Dead: the M0r - M0 CI includes zero, or M0r is above 0.0959 (gain under 0.002). Stop the refit line.
- Partial: M0r in 0.0895-0.0959. A finding to write up; no release, no further GPU.
- Strong: M0r at or below 0.0895 (half of the 0.01676 gate_up gap). Buys one follow-up: X2r = X2 with the same refit applied to UD's own gate_up. Release bar for X2r: at or below 0.0696 on S with the paired CI against UD below zero, then G4 on T1 and T2 as registered.
- Labels fixed: the "S8" screens scored all 16 chunks (KL mode ignores --chunks). The scored UD file is the mainline rebuild, 8,486,237,632 bytes, sha256-identical to Unsloth's Mainline_Support file (results/ud-mainline-sha256.md); the byte limit stays the root file's 8,486,236,352. X2 vs UD compares Metal IQ4_NL with CPU-fallback NVFP4, so "+0.00063" reads as "equal to slightly worse" (the sibling measured backend noise up to +0.00040).

## 2026-11-22, 06:50: audit corrections (Opus audit; every logged number reproduced from the logs)

- **Bug:** `gptq_q2.py --mode diag` encoded with UNIFORM channel weights, not the Gram diagonal: `np.einsum("eii->ei", gram)` is a view, so zeroing the Gram zeroed the diagonal. Fixed (`.copy()`). Consequences:
  - G2 compared GPTQ against unweighted rounding (not the matched control). GPTQ still lost (+0.00293), so the FAIL stands and is stronger.
  - `results/bias-split*.tsv` "diag" rows are unweighted rounding. Against the real diag-weighted rounding (G1, in memory: 0.11444 at L12, 0.09879 at L0, held-out conversations), UD's own Q2_0 (0.12353, 0.10905) is WORSE. That opens a cheap test, registered here: X4 = X2 with our imatrix-weighted Q2_0 (q2opt, ggml-org imatrix) at UD's 17 Q2_0 down slots. An offline check on the S-text and WikiText Grams runs first; X4 counts as an improvement only if X4 - X2 < 0 on S with the CI below zero.
- The two GPTQ v2 alpha rules disagreed (WikiText-only: 4.0; pooled: 1.0); X3 used the pooled 1.0 as registered at 05:10.
- G4 on T2 (5 chunks): in addition to the bootstrap CI, all 5 per-chunk differences must favour the candidate (one-sided sign test p = 1/32), and leave-one-out is reported.
- Fixed: the score cache key now includes the reference .kld and the binary; splice manifests no longer get overwritten; gpu_wait no longer replaces callers' traps; meanfit keeps llama-quantize's bits unless strictly better; gptq_q2 --alpha is now required.
- Omitted earlier: m0-imat-mine (our capture imatrix at M0's map) vs M0: -0.00035 [-0.00116, +0.00048].

## 2026-11-22, 07:00: M0r result

- M0r (all 24 gate_up tensors of M0 with mean-aware scale refit; modeled error -19% to -26% per layer) scores 0.17029 on S; M0r - M0 = +0.07237 [+0.06723, +0.07730], top-1 83.18% vs 87.09%. **Dead** by the 06:40 rule; the refit line stops. Like GPTQ, a lower surrogate error (here diag(Var x) + m m^T from the imatrix, in sample) went with a much worse KL. The full-Gram check on layers 6 and 20 (sibling Grams) runs next to say whether the true input-Gram error rose too.

## 2026-11-22, 07:15: attenuation hypothesis (registered before its data)

- Bug check of M0r (sibling's list): the refit decodes identically in llama.cpp's own to_float and in gguf-py (max difference 0), and it lowers plain L2, imatrix-weighted L2 and the mean-direction error. So no format bug. What it also does: it shrinks the weights. Gain <q,w>/<w,w> per layer, llama-quantize / refit / UD: L0 IQ1_S 0.944 / 0.793 / 0.947; L5 IQ2_XXS 0.974 / 0.890 / 0.976; L16 IQ2_S 0.995 / 0.941 / 0.996 (energy <q,q>/<w,w>: 1.137 / 0.804 / 1.172 at L0).
- Same pattern in the down projections: our Q2_0 encodings (q2opt, GPTQ, uniform) have gain 0.86-0.87 and energy 0.86-0.89; UD's Q2_0 has gain 0.91 and energy 0.95 (L0, L12).
- Hypothesis: least-squares fitting attenuates each tensor's output (gain < 1), and in a model with 48 layer visits that attenuation compounds instead of averaging out, so a lower MSE can mean a worse KL. UD's encodings attenuate less.
- Test (cheap, no new encoder): rescale each output row of our Q2_0 down tensors to gain 1 (multiply the fp16 block scales of that row by 1/gain; codes unchanged; `scripts/degain.py`). M0u = M0 with de-attenuated down at its 17 Q2_0 slots, vs M0. X5 = X2 with de-attenuated q2opt at UD's 17 Q2_0 slots, vs X2 (the release-relevant comparison: X5 counts only with the paired CI vs X2 below zero, and as a candidate only below UD). X4 (raw q2opt) is dropped: it attenuates more than UD.

## 2026-11-22, 07:10: attenuation test result; decomposition of the M0 gap

- X5 (X2 with q2opt down rescaled per row to gain 1 at UD's 17 Q2_0 slots) 0.07460 on S: X5 - X2 +0.00333 [+0.00217, +0.00442]; X5 - UD +0.00396 [+0.00267, +0.00524]. **Fails.** Against the nearly identical build with raw q2opt down (m0-udgu-udne, 0.07486) de-attenuation buys about 0.0003. So attenuation explains the refit failure (gain 0.79 at IQ1_S), but not our Q2_0 down gap to UD (UD itself runs at gain 0.91).
- m0-udgu-udne (UD's gate_up and UD's own non-expert tensors, our down): 0.07486. Splits the 0.01051 rest of the M0 gap: non-expert encoding 0.00629 [0.00578, 0.00679] (m0-udgu - m0-udgu-udne), down encoding 0.00422 [0.00292, 0.00549] (m0-udgu-udne - UD), of which about 0.0006 is IQ4_NL vs NVFP4 (X2).
- Full M0 gap to UD, 0.02727: gate_up 0.01676 (61%), non-expert 0.00629 (23%), down 0.00422 (15%). UD's encodings beat ours in every tensor class at equal types; no tested encoder of ours closes any part of it.

## 2026-11-22, 08:20: M0u result; X6 registered (before its data)

- M0u (M0 with q2opt down rescaled per row to gain 1 at its 17 Q2_0 slots; codes unchanged): 0.09093 on S, M0u - M0 -0.00698 [-0.00831, -0.00560]. The first encoding change of ours that helps: de-attenuating our down projections closes about a quarter of the M0 gap (M0u - UD still +0.02029 [+0.01834, +0.02224]). Inside UD's file the same fix bought about 0.0003 (X5), so the effect depends on what else in the file attenuates.
- UD's own tensors attenuate too: gain 0.91 (Q2_0 down, layers 0 and 12) and 0.95-1.00 (gate_up). Test X6, on X2 with codes unchanged and each output row's block scales multiplied by 1/gain (gain against BF16, with UD's per-expert .scale kept):
  - X6d: UD's 17 Q2_0 down projections rescaled.
  - X6dg: X6d plus all 24 gate_up tensors (UD's) rescaled.
- Reading: each vs X2 and vs UD on S, paired. A candidate only if it beats UD with the paired CI below zero; then G4 on T1 and T2 (T2 with the sign rule) before any claim.

## 2026-11-22, 08:40: release policy from Daniel (relayed); M0uu registered

- Daniel: prefer our own encodings. Credit others only if we copy 1:1 and the result improves on the original. Release paths are therefore our own encodings. X2, X6 and the donor mixes are diagnostics. An Unsloth-derived file is a candidate only if it beats UD (credited), and then it is Daniel's call. Option (b) "X2 as a repack" is dropped.
- M0uu (own encodings only): M0u with our 24 gate_up tensors (llama-quantize, M0's types) and our 7 IQ4_NL downs also rescaled per row to gain 1. Reading: M0uu vs M0u on S, paired.

## 2026-11-22, 09:15: M0rg registered (before its data)

- M0rg = M0uu with the mean-aware refit gate_up (M0r's donors, which lowered the full-Gram error in 128/128 experts at layer 6 but shrank the gain to 0.79-0.94) rescaled per row to gain 1. Reading: M0rg vs M0uu on S, paired. It isolates the refit's value once attenuation is removed.
- Small-job rule agreed with the sibling (09:05): jobs under 2 GB RSS and under 2 minutes may run beside a held CPU lock while memory_pressure shows at least 30% free. `degain2.py` now streams one expert at a time (1.4 GB RSS, about 1 min per tensor).

## 2026-11-22, 10:10: M0uu result; M0rg replaced by M0rk (before data)

- M0uu (M0u plus our gate_up and IQ4_NL down rescaled to gain 1): 0.09231, M0uu - M0u +0.00138 [+0.00076, +0.00198]. **Worse.** Gain 1 is not a universal target. For Q2_0 down (gain about 0.86, linear into the residual) it helped (M0u -0.00698). For gate_up (gain 0.94-0.99, into SwiGLU) it overshoots: energy goes from about 1.14 to 1.28 at IQ1_S.
- M0rg (refit gate_up rescaled to gain 1) is replaced, before any of its data, by M0rk: the refit gate_up rescaled per row back to llama-quantize's own per-row gain (the gain constraint), on M0u. Reading: M0rk vs M0u on S, paired. Its GPU score runs only after the sibling's two priority runs.

## 2026-11-22, 11:00: M0rk result, best own file

- M0rk (M0u with the mean-aware gate_up refit rescaled per row back to llama-quantize's per-row gain): **0.08713** on S.
  - M0rk - M0u -0.00380 [-0.00451, -0.00313].
  - M0rk - M0 -0.01078 [-0.01193, -0.00960]: 40% of the 0.02727 gap to UD closed with our own encoders only.
  - M0rk - (M0 with UD's gate_up) +0.00598 [+0.00443, +0.00759]: our gate_up now closes 64% of UD's gate_up share (0.01676).
  - M0rk - UD +0.01649 [+0.01463, +0.01820].
- Reading: the refit's lower full-Gram error was real. M0r lost only through attenuation, and restoring the reference gain keeps the gain in error and drops the harm. Rule for every encoder here: minimise output error with the row gain held at the reference encoder's.
- Sibling's cand-u (their v1 rescaled to gain 1): -0.00034 [-0.00068, +0.00003], neutral. Their file barely attenuates (IQ4_NL down at 0.996).

## 2026-11-22, 11:20: non-expert share, output-head importance; M0rko registered (before data)

- All 178 non-expert matrices of ours (llama-quantize at UD's types, pool ne-ud) match UD's in plain relative error to within 1e-5 (first 4,096 rows each), and 33 of 178 start byte-identical. Yet UD's non-expert tensors are worth 0.00629 on S.
- ggml-org's imatrix has no entry for output.weight or token_embd. So our IQ4_XS output head was quantized without importance weights. UD's was most likely weighted, and the output head feeds the logits directly.
- Test M0rko: an imatrix with `--process-output` (cand.gguf forward, 160 chunks of calib-fit), the non-expert pool requantized with it at UD's types (ne-ud2), then M0rko = M0rk with ne-ud2. Reading: M0rko vs M0rk on S, paired. Own encodings only.
- Order: the sibling's two runs first, then M0rko (own path), then the X6 diagnostics.

## 2026-11-22, 11:30: M0rko result

- M0rko (non-expert pool requantized with an imatrix that covers output.weight): 0.08749, M0rko - M0rk +0.00036 [-0.00009, +0.00084]. Neutral. Importance weighting of the IQ4_XS output head is not the non-expert lever.

## 2026-11-22, 11:45: non-expert tensors, where UD's edge is not; diagnostic registered

- All non-matrix tensors (norms, router, loop controls, convs, biases: 600+) are byte-identical between our pool and UD. Q8_0: identical codes in every block, d differs by about 1 fp16 ulp in 77% of blocks (UD's slightly worse vs BF16). So the 0.00629 non-expert share sits in the 60 lower-bit attention matrices (Q4_K-Q6_K, IQ3_XXS) and the IQ4_XS head and embedding.
- On those 60, ours beats UD in plain and in diagonal-imatrix-weighted error in 60/60 (e.g. Q5_K 0.00122 vs 0.00150). UD's edge there is invisible to any diagonal metric. Candidate explanations: (a) a Hessian-aware (off-diagonal) encoder for attention, (b) joint tuning with UD's own expert tensors (the 0.00629 was measured with UD's gate_up in place).
- Diagnostic m0rk-udne = M0rk with UD's non-expert tensors. Reading: m0rk-udne - M0rk near -0.006 means the share is real in our stack (then capture attention Grams to test (a)); much smaller means (b).

## 2026-11-22, 11:55: X6d result; X6dg dropped

- X6d (X2 with UD's own 17 Q2_0 downs rescaled per row from gain 0.88-0.91 to 1): 0.07312, X6d - X2 +0.00185 [+0.00119, +0.00248], X6d - UD +0.00248. Worse.
- Reading: gain 1 is not the target. UD's Q2_0 sits at gain 0.90 / energy 0.94 and that is near its optimum. Our q2opt sat at 0.86 / 0.87, where the rescale to 1 helped (M0u). The right target is somewhere below 1, encoding-specific. The constraint that worked (M0rk) restores the reference encoder's own per-row gain.
- X6dg (adds UD gate_up at gain 1) dropped before its data: both of its parts now predict harm (X6d here, gate_up at gain 1 in M0uu), and the GPU goes to the sibling's captures and m0rk-udne.

## 2026-11-22, 12:00: gain sweep registered (before data)

- Our q2opt Q2_0 down, rescaled per row to a fixed gain g, inside M0rk (which has g = 1.00): m0rk-g93 (g = 0.93) and m0rk-g107 (g = 1.07). Reading: each vs M0rk on S, paired. Locates the KL-optimal gain for our encoding (least-squares optimum about 0.84, UD's own encoding about 0.90).
- Donors run now under the small-job rule; builds and scores after m0rk-udne.

## 2026-11-22, 12:20: m0rk-udne result

- m0rk-udne (M0rk with UD's non-expert tensors): 0.08467, m0rk-udne - M0rk -0.00246 [-0.00327, -0.00168], against -0.00629 [-0.00679, -0.00578] for the same swap with UD's gate_up in place.
- Reading: both (a) and (b) hold. About 60% of the earlier non-expert share is joint with UD's expert tensors. About 0.0025 is a standalone edge of UD's attention/head encoding, invisible to diagonal metrics (11:45). That 0.0025 is the ceiling for full-Gram work on attention in our stack, so it's parked behind gate_up and down.
