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

## 2026-11-22, 12:35: gain sweep result

- m0rk-g93 +0.00084 [+0.00012, +0.00157], m0rk-g107 +0.00284 [+0.00218, +0.00347], both vs M0rk (gain 1.00). A parabola through the three points has its minimum at g = 0.98, only about 0.00014 below g = 1. Gain 1.00 stays the target for our Q2_0 down. Sweep closed.

## 2026-11-22, 12:45: mean-aware Q2_0 down refit, offline only

- `scripts/meanfit_q2.py` (closed-form mean-aware refit of the Q2_0 block scales, then gain restore to 1; tests/test_meanfit_q2.py checks it against a dense solve). Layer 0: surrogate error after/before 0.979, against 0.83 for gate_up at layer 6 (the M0rk lever, -0.00380 KL). The down inputs carry less mean energy (4-13% vs 16-27% for gate_up). The expected KL effect lies below the S noise, so no model run.
- Own-path status: the remaining shares (gate_up about 0.006, down about 0.003 (X5), attention about 0.0025 (m0rk-udne)) are all code-level and invisible to diagonal metrics. They need full-Gram code refinement (the sibling's refine-codes.py), which is not feasible tonight at this size: other types, other Grams, and the CPU is committed to the sibling's v2r chain.
- M0rk is frozen as the own-encoding candidate. G4 next (T1, T2 vs UD-IQ2_XXS, UD-IQ1_M, AD-IQ2_XXS-Q2; T2 sign rule), for the record.

## 2026-11-22, 13:15: score() clobbered its caller's variables (fixed)

- `score` in lib.sh assigned plain `name`, `model`, `set`, `log`, `ref`, `sig`, `ctx`. In `comparators.sh` with CAND, the loop variable `name` was overwritten after the T1 run, so the T2 run asked for `models/ext/final-UD-IQ2_XXS-T1-P3.gguf` and died at 12:53, leaving a stale GPU lock (removed). Fix: `score`'s variables are `_s`-prefixed, and a failed run releases the GPU lock and returns 1.
- Earlier results unaffected: the only caller that reads one of these names after a `score` call is the CAND branch of comparators.sh, first run today. S comparator runs called `score` once per iteration.

## 2026-11-22, 13:55: G4 on the frozen own-encoding candidate M0rk: FAIL vs Unsloth

- M0rk (8,486,214,944 bytes, sha256 a62fd9cc49e9..., results/m0rk.sha256), paired vs each comparator (results/release-test.txt):

  | vs | T1 diff [95% CI] | T2 diff [95% CI] |
  |---|---|---|
  | UD-IQ2_XXS (8,486,237,632 B) | +0.01424 [+0.01249, +0.01593] | +0.01662 [+0.01414, +0.01941] |
  | UD-IQ1_M (8,024,442,304 B) | +0.00411 [+0.00193, +0.00616] | +0.01628 [+0.01464, +0.01829] |
  | AD-IQ2_XXS-Q2 (7,900,679,776 B) | -0.04051 [-0.04712, -0.03499] | -0.02099 [-0.02523, -0.01776] |
  | AD-IQ2_M (8,665,091,680 B, context) | -0.01555 [-0.01871, -0.01278] | -0.00822 [-0.01193, -0.00498] |

- KL on T1 / T2: M0rk 0.08007 / 0.05934, UD-IQ2_XXS 0.06583 / 0.04272, UD-IQ1_M 0.07596 / 0.04306, AD-IQ2_XXS-Q2 0.12058 / 0.08033, AD-IQ2_M 0.09562 / 0.06756.
- Reading: the S ranking holds on held-out text. Our own encoding beats both AtomicChat files (one of them larger than ours) and loses to both Unsloth files, including the 0.46 GB smaller UD-IQ1_M. No release candidate. Option (a): publish the findings.

## 22 Nov 2026, 14:20: sibling's v2r on S (relayed, not our run)

- v2r (sibling, own encoding, full-Gram code refinement of IQ2_S / IQ3_XXS gate_up, 11,566,703,648 B, 0 tensors off Metal): S 0.02102. v2a 0.02780, UD-IQ3_XXS 0.02050.
- Paired, 16 chunks: v2r - v2a -0.00677 [-0.00758, -0.00610]; v2r - UD +0.00052 [-0.00030, +0.00133], not resolved. 93% of the gap closed. Grams come from WikiText-2 train, the same source as S, so held-out T1/T2 is still open.
- Bearing on us: M0rk's gate_up is IQ1_S (layers 0-4, 6), IQ2_XXS (5, 7, 8, 12-14, 18), IQ2_XS (9-11, 15, 17, 22) and IQ2_S (16, 19-21, 23). The refiner supports only the 5 IQ2_S layers. At the sibling's size, refinement reached parity with Unsloth, not a win, so at 8.49 GB the best case is parity too.

## 22 Nov 2026, 14:40: sibling's v2r held out (relayed), and the domain check for M0rk

- v2r - UD-IQ3_XXS: T1 +0.00138 [+0.00059, +0.00218], T2 +0.00468 [+0.00333, +0.00626]. v2r - v2a: T1 -0.00385 [-0.00470, -0.00298], T2 -0.00100 [-0.00186, +0.00009]. Gap share closed: S 0.93, T1 0.74, T2 0.18. The full-Gram refinement mostly fits its calibration domain (WikiText-2 train, about 5k tokens per median expert over 2304 dimensions). Not a release.
- M0rk's refit is not a full-Gram fit. It uses only per-expert E[x] and Var x (2 x 2304 numbers) from ggml-org's imatrix (`scripts/m0r.sh`), not from our calibration text, so it has far fewer fitted statistics to overfit.
- Held-out check, inferred (M0 was never scored on T1/T2, so no paired number): the gap from M0rk to UD-IQ2_XXS is +0.0165 on S (0.02727 - 0.01078), +0.01424 on T1 and +0.01662 on T2. The gap does not widen off S, so nothing points to M0rk's gain being S-specific.

## 22 Nov 2026, 15:00: correction to the release bar

- concept-v2 and the 14:20 entry applied "must clearly beat UD" to our own encodings. The policy (Daniel, 22 Nov) applies that bar to derivative builds that carry others' tensors. An own encoding needs no win over UD. The sibling's v1 went public on its Mac track (lowest KL among all-Metal files under its size) after failing its UD gate.
- Consequence: M0rk (8.49 GB, all-Metal, own encoding) is a possible release on the same frame, if it is the closest-to-original all-Metal file at or under its size. That still needs the remaining comparators and a speed run. Daniel decides.
- New tier under study: <= 9,138,715,328 B (UD-Q2_K_XL). UD-Q2_K_XL's header: 11 NVFP4 down tensors (2.10 GB) plus 1 MXFP4; gate_up IQ2_S in 11 layers (9, 11, 15-23).

## 22 Nov 2026, 15:40: concept v3 (16 GB Mac file, sized by measurement)

- Reviews of the 9 GB draft: Fable, Astra and Sol (`reviews/2026-11-22-*-9gb-*.md`); census of 27 same-model GGUFs at 7.5-9.6 GB (`results/landscape-7.5-9.6GB.md`); literature pass (`reviews/2026-11-22-research-levers-sonnet.md`). No model run.
- New facts:
  - Stock Metal rejects only NVFP4, TQ1_0 and F8_E4M3 for mul_mat (upstream c66103334).
  - Upstream 690b87d fixes the pass-2/3 scales for UD files.
  - The only unscored all-Metal competitor at 8.67-9.14 GB is GSQ-RCO-IQ2_XS (8,949,801,504 B).
  - Exact promotion costs per layer: down Q2_0→IQ4_NL +95,551,488 B; gate_up IQ1_S→IQ2_XXS +42,467,328 B, IQ2_XXS→IQ2_XS and IQ2_XS→IQ2_S +21,233,664 B each, IQ2_S→IQ3_XXS +42,467,328 B.
- Open conflict: PLAN.md's measured 2.4 GB at 16K (cap about 9.05 GB) against Sol's KV-only estimate of 0.5-1.3 GiB at 8K-32K. Step 2 measures it.
- Plan: M0rk as base, equal-byte promotions ranked by KL per byte, per-block gain sweep, refiner pilot. GPU steps wait for windows the sibling grants.

## 22 Nov 2026, 19:40: pre-registration for concept-v3 steps 1-3 (before any step-2 or step-3 result)

- Workload for the fit claim: one text stream, P = 3, f16 KV, default macOS GPU budget of 11,453 MB, at 8K and 16K context. The size cap is set after step 2's memory run: the largest size whose projected peak at 16K (M0rk's measured overhead added to the weights) stays under 11,453 MB, and never above 9,138,715,328 B.
- Comparators for the Mac track (same weights, every expert tensor in a Metal type, at or under the candidate's size), scored on S, T1 and T2 against the same Q8_0 reference:
  - AtomicChat AD-IQ2_XXS-Q2 7.90 GB (done), AD-IQ2_M 8.67 GB (done), AD-IQ1_M 7.26 GB and AD-IQ1_S 6.88 GB;
  - abenzerps GSQ-RCO-IQ2_XS 8.95 GB.
  - Context only, not all-Metal: UD-IQ1_S 7.76 GB (1 NVFP4), UD-IQ1_M, UD-IQ2_XXS, UD-Q2_K_XL. Different weights (abliterated, heretic) are out.
- Gate per comparator: paired CI of candidate minus comparator below 0 on T1 and T2, and top-1 not lower. M0rk's existing T1/T2 runs count for M0rk. A new candidate gets T1/T2 once frozen, then T3.
- Step 3, single promotions on M0rk, scored on S (16 chunks, paired against M0rk), each 85-96 MB:
  - p-d13: down 13 Q2_0 -> IQ4_NL (pool-3bit), +95,551,488 B.
  - p-d21: down 21 Q2_0 -> IQ4_NL, +95,551,488 B.
  - p-g46: gate_up 4 and 6 IQ1_S -> IQ2_XXS (pool-2bit, mean-aware refit plus gain restore as M0rk), +84,934,656 B.
  - p-g1318: gate_up 13 and 18 IQ2_XXS -> IQ2_S (gu-iq2_s, refit plus gain restore), +84,934,656 B.
  - p-g2023: gate_up 20 and 23 IQ2_S -> IQ3_XXS (pool-3bit, plain; meanfit has no IQ3_XXS), +84,934,656 B.
  - Rank by (KL change) / (bytes added). Combine the promotions whose CI is below 0, best ratio first, up to the cap, and verify the combination on S. Stop rule: if none of the five has a CI below 0, step 3 ends and the candidate stays M0rk.

## 22 Nov 2026, 20:50: outside evidence on scale training (relayed by the sibling, source read)

- llama.cpp discussion #31320 (Marco-Nano-Instruct, 8B MoE, 232 experts; self-report). It trains only the block scales with codes frozen, against forward KL to bf16 on 24-50M tokens. Output is the same stock types. KL vs llama.cpp's imatrix files: IQ3_S 0.0277 vs 0.0353, IQ2_S 0.0566 vs 0.0963, IQ2_XXS 0.0914 vs 0.173. Not stated: optimizer, hardware, wall time, calibration domain, code URL.
- Bearing: the strongest outside evidence that scale-only fits in stock IQ types carry large KL gains, and it supports our refit direction. End-to-end training of a 26B model over tens of millions of tokens does not fit a 48 GB Mac without a GPU in days (inferred). A middle path would fit scales per layer against the layer output on captured inputs (local, like ours, but with the trained loss). Parked until steps 3-4 report.
- Also relayed: MACaQ (arXiv 2611.10649 v2) Appendix C publishes full-Hessian codes-only IQ3_S / IQ3_XXS search, so any write-up of code refinement cites it. hridya423 mixB pads down rows to 1280 and needs a fork, so it is not a comparator.

## 23 Nov 2026, 00:55: step 2 memory (measured) and the size cap

- `/usr/bin/time -l` around one `llama-bench -ngl 99 -p 512 -n 128 -d D -r 1` (flash attention auto, f16 KV), build fcaddf97a. Run with the GPU lock only, beside the sibling's refiner, so the tg numbers at depth are not clean speed numbers (bench2 later).

  | file | bytes | depth | max RSS | peak footprint | tg128 at depth |
  |---|---|---|---|---|---|
  | M0rk | 8,486,214,944 | 8K | 8,880,635,904 | 1,460,346,512 | 24.08 |
  | M0rk | 8,486,214,944 | 16K | 10,911,596,544 | 2,477,122,008 | 21.56 |
  | M0rk | 8,486,214,944 | 32K | 11,984,437,248 | 3,515,262,168 | 18.14 |
  | GSQ-RCO-IQ2_XS | 8,949,801,504 | 16K | 10,577,281,024 | 2,478,007,032 | 20.15 |
  | UD-IQ2_XXS | 8,486,237,632 | 16K | 10,384,900,096 | 2,574,426,680 | 14.99 |
  | UD-Q2_K_XL | 9,138,716,608 | 16K | 11,272,552,448 | 2,573,968,336 | 13.21 |

- Reading: peak footprint (memory not backed by the model file: KV, compute and scratch) is the same for all-Metal files of this model at a given depth, 2.48 GB at 16K. It grows about 1.03 GB per extra 8K-16K of context. Max RSS counts only touched pages of the mapped weights, so it is not comparable across files.
- Size cap by the pre-registered rule (weights plus M0rk's measured 16K overhead under 11,453,250,000 B): 11,453,250,000 - 2,477,122,008 = **8,976,127,992 B**. Headroom over M0rk: 489,913,048 B. GSQ-RCO-IQ2_XS (8,949,801,504 B) is under it. UD-Q2_K_XL is 162,588,616 B over, but 2.10 GB of its weights run on the CPU, so the GPU budget does not bind it the same way (it uses about 11.71 GB of system memory at 16K, inferred from bytes plus footprint).
- At 32K, M0rk needs about 12.0 GB (8.49 + 3.52), over the default GPU budget. So the 16 GB claim is 16K, not 32K.

## 23 Nov 2026, 01:35: step 2 KL for the 9 GB comparators (measured, results/step2/compare.txt)

- KL S / T1 / T2: UD-Q2_K_XL 0.05546 / 0.05104 / 0.03492. GSQ-RCO-IQ2_XS 0.09389 / 0.08898 / 0.05573. M0rk 0.08713 / 0.08007 / 0.05934.
- M0rk minus GSQ-RCO-IQ2_XS: T1 -0.00891 [-0.01101, -0.00685], top-1 88.68% vs 88.29%. T2 +0.00361 [+0.00050, +0.00630], top-1 91.44% vs 91.69%. M0rk minus UD-Q2_K_XL: T1 +0.02903, T2 +0.02442.
- AD-IQ1_M: S 0.18241, T1 0.17574, T2 0.12621.
- Reading: GSQ-RCO is 463,586,560 B larger than M0rk and loses on prose but wins on reasoning traces. A 9 GB candidate (cap 8,976,127,992 B) must beat GSQ-RCO on T2 as well. M0rk's own Mac-track set (at or under 8.49 GB) does not contain GSQ-RCO.

## 23 Nov 2026, 01:40: amendment to step 3 (before any promotion result)

- Because GSQ-RCO wins on T2 (01:35), each single promotion is also scored on T2, used as development data, paired against M0rk's T2 run (final-cand-T2-P3).
- A promotion enters the combination only if its S CI is below 0 (as registered) and its T2 point estimate is below 0. The ranking stays by S KL per byte.
- The combination's cap is 8,976,127,992 B (00:55). The frozen candidate still gets T1, T2 and a fresh T3.

## 23 Nov 2026, 03:35: Mac-track comparators at or under M0rk's size (measured, results/step3/compare-iq1.txt)

- KL S / T1 / T2: AD-IQ1_M 0.18241 / 0.17574 / 0.12621; AD-IQ1_S 0.21339 / 0.20504 / 0.14737; UD-IQ1_S (7,764,293,056 B, 1 NVFP4) 0.09214 / 0.08880 / 0.05194.
- M0rk minus, T1 / T2:
  - AD-IQ1_M -0.09567 / -0.06687
  - AD-IQ1_S -0.12497 / -0.08803
  - UD-IQ1_S -0.00873 [-0.01196, -0.00621] / **+0.00740 [+0.00513, +0.00935]**
  - with AD-IQ2_XXS-Q2 from 22 Nov: -0.04051 / -0.02099
- **Mac-track gate at M0rk's size: PASS.** M0rk beats every same-weights all-Metal file at or under 8,486,214,944 B (AD-IQ2_XXS-Q2, AD-IQ1_M, AD-IQ1_S) on T1 and T2 with CIs below 0 and higher top-1. Speed: 1.83x / 1.52x (pp512 / tg128) against UD-IQ2_XXS (21:15 bench). Release is Daniel's decision, and the v1 frame also asks for T3, the stability checks and an unpatched-upstream score first.
- The weakness is reasoning traces: on T2, UD-IQ1_S (0.72 GB smaller), UD-IQ1_M and GSQ-RCO all beat M0rk, while M0rk wins T1 against UD-IQ1_S and GSQ-RCO. Unsloth calibrates on conversation data, and GSQ-RCO on a 2.05M-token set of its own. Our refit read ggml-org's imatrix. The night's finding that the imatrix does not matter (< 0.0004) was measured on S, prose only. Open: does calibration domain move T2?

## 23 Nov 2026, 03:45: pre-registration, step 3c (does the imatrix domain move T2?)

- Diagnostic, development data only. Rebuild three files that differ only in the imatrix of their gate_up codes:
  - M0: ggml-org's imatrix.
  - m0-imat-mine: our capture on calib-fit, prose plus 137 conversations.
  - m0-imat-bart: bartowski's imatrix.
  - All three keep the same downs and non-expert tensors. On S they differed by less than 0.0004.
- Score each on T2 and pair against M0.
- Reading:
  - If a variant differs from M0 on T2 by more than 0.003 with its CI excluding 0, the imatrix domain is a lever for reasoning traces. The next step is then an imatrix that includes model-generated reasoning traces, disjoint from T2's GSM8K items.
  - If both are within ±0.002, the T2 gap is the encoder's, as on S, and the domain route is dropped.

## 23 Nov 2026, 04:10: M0rk on unpatched upstream llama.cpp (measured)

- S with the sibling's clean upstream build c661033 (contains 690b87d): 0.08713, top-1 87.91%. All 16 per-chunk KLs are identical to our patched build fcaddf97a. The score signature records the upstream binary's mtime, so the cache was not hit. M0rk carries no `.scale` tensors, and the loop patches touch no code path it uses.
- v1-frame checks still open for M0rk: T3, the 7 stability checks, a load on LM Studio.

## 23 Nov 2026, 04:40: stability battery on M0rk (measured, results/stability/verdict-m0rk.md)

- The sibling's battery and thresholds, unchanged, judged against its Q8_0 reference run. **PASS, 7 of 7.** Lowest distinct 4-gram ratio 0.963 (reference 0.9894). Longest repeated-line run 1. Planted fact found at 16K. Tool calls 5 of 5. Non-English 2 of 2. GSM8K 30 of 30, 0 capped (a smoke test, not an accuracy estimate).

## 23 Nov 2026, 06:15: pre-registration, step 3d (M0rk's refit on mixed-domain statistics)

- Why: the sibling's amendment 13 (b1a8e51 in its repo) moved its refiner's T2 share of the gap from 0.18 to 0.70 by changing only the calibration to `data/calib-mix.txt`. That file is WikiText-2 train, chat-templated conversations and prose, 300 x 512 tokens, with --parse-special. M0rk's refit reads per-expert E[x] and Var x from ggml-org's imatrix.
- Build, M0rk-mix:
  - Capture an imatrix (our llama-imatrix, which stores in_sum and in_sum2) on the sibling's cand.gguf at P = 3, on the sibling's calib-mix.txt: --chunks 300 -c 512 --parse-special. The file has 0 shared 13-grams with S, T1, T2 and T3 (sibling `results/data.md`).
  - Then M0rk's recipe unchanged: mean-aware refit of the same llama-quantize gate_up pools, then each row back to its pool's gain. Same downs and non-expert tensors.
  - Only the refit's statistics change, and the codes stay those of M0rk, so the run isolates the domain of the statistics. Size equals M0rk's.
- Reading, paired against M0rk on S, T1 and T2 (all seen, development):
  - Adopt the mixed statistics if T2 has its CI below 0 and S does not rise by more than 0.001.
  - If T2's CI includes 0, the domain of the refit statistics is not the lever for M0rk's T2 gap. The next suspect is then the domain of the codes' imatrix (step 3c).

## 23 Nov 2026, 06:40: quiet speed re-bench (measured, results/step2/bench2-*.txt)

- Both locks held, the four files back to back, `llama-bench -ngl 99 -p 512 -n 128 -r 3`, build fcaddf97a, M5 Pro.

  | file | bytes | pp512 t/s | tg128 t/s |
  |---|---|---|---|
  | M0rk | 8,486,214,944 | 252.71 ± 13.99 | 19.08 ± 0.24 |
  | GSQ-RCO-IQ2_XS | 8,949,801,504 | 192.61 ± 14.25 | 18.67 ± 0.26 |
  | UD-IQ2_XXS (7 NVFP4 on CPU) | 8,486,237,632 | 128.68 ± 0.23 | 12.44 ± 0.33 |
  | UD-Q2_K_XL (11 NVFP4 on CPU) | 9,138,716,608 | 92.49 ± 1.60 | 11.78 ± 1.38 |

- M0rk vs UD-IQ2_XXS: 1.96x prompt, 1.53x generation. vs UD-Q2_K_XL: 2.73x and 1.62x. vs GSQ-RCO: 1.31x prompt (its Q8_0 attention), generation about equal.

## 23 Nov 2026, 07:15: step 3c result (measured, results/step3/t2-imatrix.txt)

- T2: M0 0.06517. m0-imat-mine 0.06517, diff -0.00000 [-0.00200, +0.00212]. m0-imat-bart 0.06447, diff -0.00070 [-0.00169, +0.00014].
- By the registered rule (both within ±0.002): the domain of the imatrix behind llama-quantize's gate_up codes does not move T2, as it did not move S. Our capture included 137 conversations. The domain route for the codes is dropped.
- Step 3d is still open. It tests the domain of the refit's E[x] and Var x, which enter the mean term that the diagonal imatrix weight ignores.

## 23 Nov 2026, 07:50: step 3d result (measured, results/step3/m0rk-mix.txt)

- M0rk-mix (M0rk's recipe, refit statistics from a calib-mix imatrix, same bytes) minus M0rk:
  - S -0.00288 [-0.00386, -0.00197], top-1 88.15% vs 87.91%.
  - T1 -0.00224 [-0.00293, -0.00162], top-1 88.84% vs 88.68%.
  - T2 -0.00025 [-0.00185, +0.00158], top-1 91.55% vs 91.44%.
- Registered rule: T2's CI includes 0, so the adoption criterion is not met. Domain of the refit statistics is not the lever for the T2 gap. With step 3c, calibration domain is ruled out for our recipe on T2, both for the codes' imatrix and for the refit statistics. What remains is the encoder's code choice, the same edge UD holds on S.
- Post hoc decision, labeled as such (Daniel may reverse it): M0rk-mix replaces M0rk as the base for the step-3 combination. It is better on S and T1 with CIs below 0, not worse on any point estimate, and costs no bytes. The promotions were measured on M0rk. Their effects are assumed to carry over, and the combination is verified on its own.

## 23 Nov 2026, 08:00: step 3 result and the combination (measured, results/step3/promotions.txt)

- All five single promotions on M0rk have S CIs below 0 and T2 point estimates below 0, so all enter by the registered rules (19:40 and 01:40). Ranked by S change per 100 MB:
  - p-g46: -0.00353 (S -0.00300 [-0.00406, -0.00189], T2 -0.00207 [-0.00410, +0.00055])
  - p-d21: -0.00288 (S -0.00275, T2 -0.00092)
  - p-g1318: -0.00246 (S -0.00209, T2 -0.00114)
  - p-g2023: -0.00239 (S -0.00203, T2 -0.00071)
  - p-d13: -0.00215 (S -0.00205, T2 -0.00181)
- Together they add 445,906,944 B, which fits the cap (8,976,127,992 B).
- Combination c1, on the M0rk-mix base (07:50, post hoc):
  - gate_up 4 and 6 IQ2_XXS and 13 and 18 IQ2_S, each refit with the mix statistics plus gain restore as M0rk-mix;
  - gate_up 20 and 23 IQ3_XXS plain;
  - down 13 and 21 IQ4_NL;
  - everything else as M0rk-mix.
  - Expected 8,932,121,888 B. Scored on S, T1 and T2 against M0rk-mix (verification) and against the comparators.
- Reading: c1 is the 9 GB-tier development candidate if every promotion's share holds (c1 minus M0rk-mix has a CI below 0 on S and T2). It is frozen only after that, then T3 (Daniel).

## 23 Nov 2026, 08:20: c1 result (measured, results/step3/c1.txt, c1-compare.txt)

- c1: 8,932,121,888 B, sha256 9975a56dbf2e2bcd91b7c56ff98ef59f0840cdc51d29a8980c0f67b4707531b0, every tensor in a Metal type. KL S / T1 / T2 = 0.07209 / 0.06683 / 0.04889, top-1 89.03% / 89.67% / 92.14%.
- Verification against the base M0rk-mix: S -0.01216 [-0.01354, -0.01093], T1 -0.01100 [-0.01252, -0.00949], T2 -0.01020 [-0.01222, -0.00739]. The five promotions add up, as the sum of the single effects predicted (about -0.012 on S).
- **Mac track at the 9 GB tier: PASS.** c1 beats every same-weights all-Metal file at or under the cap on T1 and T2, with CIs below 0 and higher top-1.
  - GSQ-RCO-IQ2_XS (8,949,801,504 B, larger than c1): T1 -0.02215, T2 -0.00684.
  - AD-IQ2_M: T1 -0.02879, T2 -0.01867.
  - AD-IQ2_XXS-Q2, AD-IQ1_M, AD-IQ1_S: T1 -0.054 to -0.138, T2 -0.031 to -0.098.
- Context, not all-Metal:
  - UD-IQ2_XXS (8.49 GB): T1 +0.00100 [-0.00092, +0.00280] (tie), T2 +0.00617.
  - UD-IQ1_M: T1 -0.00913, T2 +0.00583.
  - UD-IQ1_S: T1 -0.02197, T2 -0.00305 with top-1 92.14% vs 92.71%.
  - UD-Q2_K_XL: T1 +0.01579, T2 +0.01397.
- T1 and T2 were used in development (the T2 promotion amendment, the M0rk-mix base decision), so these are development readings. A release claim needs T3 (Daniel).
- c1 is frozen as the 9 GB-tier candidate at this sha256. Next, the v1-frame checks on c1: peak memory at 16K, the unpatched-upstream S score, the stability battery and the quiet speed bench.

## 23 Nov 2026, 08:35: pre-registration, step 5 (full-Gram code refinement of c1's supported gate_up layers)

- Why: the sibling's v2r to v2m moved T2 by -0.00299 through full-Gram code search on mixed-domain Grams. Our diagonal refit statistics on the same text did not (step 3d). c1 has gate_up in the refiner's types in 7 layers: IQ2_S at 13, 16, 18, 19 and 21, IQ3_XXS at 20 and 23.
- Build, c1r:
  - Capture full per-expert input Grams of those 7 layers with our llama-imatrix (GRAM_DIR/GRAM_MATCH) on the sibling's cand.gguf at P = 3. Text: calib-mix.txt, 300 x 512 tokens, --parse-special, as the sibling's amendment 13. Two groups (13 16 18 19) and (20 21 23).
  - Refine with the sibling's `scripts/refine-codes.py`, used read-only from its repo, at its v2m settings: shrink 0.1, min tokens 2304, gain guard on, sweeps 4, tol 1e-4, 4 workers. --quant is c1, so each row's gain floor is c1's own.
  - Codes change, scales stay. Splice the 7 refined tensors into c1. Same size.
- Reading, paired against c1 on S, T1 and T2 (development):
  - Accept c1r if T2 has its CI below 0 and neither S nor T1 rises by more than 0.001.
  - Otherwise c1 stays the candidate, and refinement at 7 of 24 layers is recorded as too small for this file.
- Credit: the refiner and the calibration text are the sibling's, our own work in Daniel's repos (no outside credit needed). The write-up cites MACaQ (arXiv 2611.10649, Appendix C) for full-Hessian codes-only search in IQ formats.

## 23 Nov 2026, 08:45: step 5 provenance and release rule

- Refiner used read-only: sibling repo at 7a4d0de, scripts/refine-codes.py sha256 1991258126ea616dd0e1f487405bc0f6cf019ea3e4fbf4235569c991e2e4a0f0, its helper scripts/gram-error.py sha256 c02e74d384fae116aac9e5b7d816411c1e4487a74180412ff44be7cbadbf1760. Uncommitted changes to the two files: 0. The sibling will not edit them during the run.
- Release rule: Daniel has not reviewed refine-codes.py yet. No result from it counts toward a release before that review, so c1r can at most become a candidate pending review. c1 is not affected.
- Known caveat (sibling, inferred): the Grams come from cand.gguf's forward, not c1's. c1's early layers are IQ1_S, so its later layers see inputs further from cand's than v2a's did. If c1r disappoints, check this mismatch first.

## 23 Nov 2026, 09:00: c1 release-frame checks (measured, results/c1/)

- Peak memory (`/usr/bin/time -l`, llama-bench at depth):
  - 16K: footprint 2,476,008,208 B. Weights plus footprint 11,408,130,096 B, under the 11,453,250,000 B default GPU budget of a 16 GB Mac by 45 MB.
  - 32K: footprint 3,528,091,176 B, over the budget. The fit claim is 16K.
- Unpatched upstream c661033: S 0.07209, identical to our build.
- Stability battery: **PASS, 7 of 7.** 4-gram ratio 0.9791 (reference 0.9894), repeat run 1, fact found, tools 5/5, non-English 2/2, GSM8K 30/30 with 0 capped.
- Speed, c1 and M0rk back to back with both locks: c1 pp512 267.99 ± 2.85, tg128 21.61 ± 0.03. M0rk 268.56 ± 2.52, 21.90 ± 0.10. The 446 MB of promotions cost no measurable speed. Against UD-IQ2_XXS's quiet bench (06:34, 128.68 / 12.44), c1 is about 2.1x / 1.7x, from different sessions.
- Open for a release of c1: T3 (Daniel's approval of the sibling's amendment 1 tokens, or a fresh set of our own), Daniel's release decision, and a load test in LM Studio.

## 23 Nov 2026, 10:30: code review by Astra and Fable, fixes (reviews/2026-11-23-code-pipeline-*.md)

- Daniel's request, in his words: "let Astra and Fable do the code review and then ask me if something is still left. no hard release limitations bound to my code review."
- Neither reviewer found a bug in the numerical core that changes the bytes we built. Their findings are in resume, caching, provenance, locks and edge cases.
- Fixed here:
  - score(): the key binds model and reference (with inode), the dylibs and the argument boundaries, and is re-checked after gpu_wait; identity arguments are refused; the log is published atomically with validation.
  - kl.tsv: a model_state column, and compare.py leaves out stale runs.
  - assemble.py: an atomic write, an alias guard, a manifest check for freshness, and refusals for duplicate layers, ne overlaps and dropped .scale.
  - degain2.py: the reference .scale, target checks, rows left alone counted, the reached gain recomputed, BF16 asserted.
  - meanfit.py: exact level scan, acceptance from decoded residuals, idle coordinates kept, scale-bit-only asserted per expert, scaled sources refused.
  - tests/test_meanfit.py: Astra's two counterexamples.
- Checked:
  - Headline files hash to their recorded sha256, and scores are signed against them.
  - One gap: m0rk-S was signed against an earlier write of m0rk.gguf (re-assembled 22 Nov 12:38, not logged). Rescore running.
  - The fixed meanfit gives the same error ratio on real data (6,448 bytes of 132.7M differ), so c1 is not rebuilt.
  - The fixed degain2 reproduces M0rk's donor byte for byte.
- Open:
  - The lock race (both repos, proposed to the sibling).
  - KL rounding in per-chunk reconstruction (bounded, no verdict flips).
  - A content digest for references.
  - The findings in the sibling's refine-codes.py, gram-error.py and splice.py, sent to it.

## 23 Nov 2026, 10:40: m0rk-S rescored on the current file (measured, results/m0rk-rescore.txt)

- S 0.087126 on the current m0rk.gguf (sha256 a62fd9cc...), all 16 per-chunk values identical to the run on the 22 Nov 10:46 write. Model state ok. No stale run is left in kl.tsv.

## 23 Nov 2026, 11:15: pre-registration, off-domain check on the sibling's battery A (before any of our scores on it)

- Why: the sibling's amendment 15 found UD's lead grows off the calibration domain (v2m/UD 1.20-1.30 on code, German and maths texts, against 1.03 on T1 and 1.12 on T2). We never used these texts in development, so they are a fresh check that needs no thesis-reserved tokens.
- Data and logits: the sibling's data/battery/A-{python,rust,german,math}.txt (sha256 in its results/battery/data.md; 8 documents x 4,096 tokens each) and its Q8_0 reference logits batA-<d>-P3.kld. It regenerates them one domain at a time under its own approval, and we read them only. Settings are the sibling's: -c 1024 --chunks 32, P = 3.
- Files: c1, M0rk, c1r if built by then, GSQ-RCO-IQ2_XS, AD-IQ2_M, UD-IQ2_XXS, and UD-Q2_K_XL as context.
- Readings, paired per chunk within each domain:
  1. The Mac-track claim for c1 holds off-domain if c1 minus GSQ-RCO-IQ2_XS and c1 minus AD-IQ2_M have CIs below 0 in every domain. Otherwise the claim is narrowed to the domains where it holds.
  2. If c1r passes its T2 rule, it is adopted only if c1r minus c1 is not above 0 (CI) in any domain.
  3. Descriptive: c1/UD-IQ2_XXS and c1/UD-Q2_K_XL per domain, against the same ratio on T1 and T2.

## 23 Nov 2026, 16:25: step 5 result, c1r (measured, results/c1r/)

- c1r: c1 with full-Gram refined codes at gate_up 13, 16, 18-21, 23. 8,932,121,888 B (same as c1), sha256 c4d2f4e4...
- Paired against c1 (development data):
  - S 0.06949 vs 0.07209, -0.00260 [-0.00355, -0.00168].
  - T1 0.06561 vs 0.06683, -0.00122 [-0.00196, -0.00055].
  - T2 0.04912 vs 0.04889, +0.00023 [-0.00072, +0.00124]; top-1 91.87% vs 92.14%.
- **Rule: c1r fails.** T2's CI is not below 0. c1 stays the 9 GB-tier candidate. Recorded per the rule: refinement at 7 of 24 layers is too small to move T2 in this file.
- What it shows (inferred): the refiner lowers KL on S and T1, the WikiText-like sets, but not on T2. This is the same split as the diagonal statistics in step 3d. The caveat from 08:45 (Grams from cand.gguf's forward, not c1's) is untested.
- Battery A (registered before any of its scores, which do not exist at this time): batA.sh still scores c1r. Its c1r - c1 rows are descriptive only and cannot reverse the rule above. They tell whether the S/T1 gain carries to code, German and maths, which decides whether a refiner run on c1's own forward is worth it.
- The sibling may now edit refine-codes.py, gram-error.py and splice.py.

## 23 Nov 2026, 16:50: c1 rebuilt with the sibling's reviewed splice.py (measured, results/c1/splice-recheck.txt)

- splice.py at the sibling's ca5b1e4 (code-review fixes: empty takes exit, atomic write, raw-byte verify) rebuilds c1 from configs/c1.json to the same sha256, 9975a56d... MATCH. Our callers need no change.
- Note: splice.py now writes a side JSON named after its output path, so assemble.py's OUT.part leaves OUT.part.json. Harmless, not renamed yet.

## 23 Nov 2026, 18:25: battery A off-domain result (measured, results/batA/compare.txt)

- 7 files x 4 domains, 32 chunks of 1,024 tokens each, P = 3, the sibling's Q8_0 logits. All 28 runs are signed against the current model files.
- Reading 1, the Mac-track claim: **holds in every domain.** c1 minus GSQ-RCO-IQ2_XS: python -0.01058, rust -0.01041, german -0.01959, math -0.00926. c1 minus AD-IQ2_M: -0.01709, -0.01072, -0.02900, -0.01757. All CIs are below 0.
- Reading 2: not applicable, because c1r failed its T2 rule. Descriptive: c1r minus c1 is +0.00064, +0.00074, -0.00056 and -0.00048, every CI spanning 0. The refiner's S/T1 gain does not reach code, German or maths, so a refiner rerun on c1's own forward is not pursued.
- Reading 3, descriptive, c1/UD ratios:

  | | T1 | T2 | python | rust | german | math |
  |---|---|---|---|---|---|---|
  | c1/UD-IQ2_XXS | 1.015 | 1.144 | 1.058 | 1.027 | 1.076 | 1.129 |
  | c1/UD-Q2_K_XL | 1.309 | 1.400 | 1.335 | 1.276 | 1.324 | 1.386 |

  The gap does not grow off-domain for c1. The sibling's v2m was at 1.20-1.30 against UD there (its amendment 15). Inferred: v2m's off-domain loss comes from its own calibration-fitted encodings, which c1 does not share (c1 is ggml-org imatrix plus mean-aware refit). Untested.
- c1 to UD-IQ2_XXS: python and rust within noise (CIs span 0), german and math UD better. The claim stays the Mac track, with the gap to Unsloth stated.

## 23 Nov 2026, 18:45: shared locks move to kernel locks (lockf), paired with the sibling's 9b03ba5

- Why: the code reviews found a race in the pid-file locks. A stale-lock rm could delete a lock another job had just created with set -C, so two jobs could hold the GPU.
- scripts/lib.sh: the sibling's _take/_give. A background `lockf -k` holds the flock and watches our pid, and the kernel drops the lock when the job ends. Lock files stay on disk, and their existence means nothing. gpu_wait still waits out a running llama-bench. New both_try takes both locks only if both are free right now, else neither (bench2.sh and c1-checks.sh use it instead of their -e tests, which would now wait forever).
- tests/locks.sh (with tests/lockhold.sh), 12 checks on scratch files, all pass. Covered: exclusion, try, re-entry, retake after give, release on kill -9, both_try (busy, free, GPU already ours), exclusion against a holder taken with the sibling's own gpu-lock.sh, and no holders left over.
- No job ran on either side during the switch.
