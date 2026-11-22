# Crosscheck: maths (meanfit.py scale refit)

- Date: 2026-11-22
- Checkpoint: maths (code and algebra review)
- Model: gpt-6.2-astra, effort high
- Input: scripts/meanfit.py at 753b1f7 plus ggml-common.h block structs and gguf-py decoders (prompt in the session scratchpad)

## Reply (verbatim)

1. **IQ1_S scale bits.** `qh = (qh & np.uint16(0x8FFF)) | (ls.astype(np.uint16) << 12)` is correct for the stated little-endian layout. The 50-byte block has eight 16-bit words starting at byte 34. Bits 12–14 select \(l=0,\ldots,7\) for 32 weights, with scale \(d(2l+1)\). The mask preserves both the low 12 grid-index bits and bit 15, which selects the sign of the \(1/8\) offset *inside* the scaled code vector. Keep the mask and multiplier. Validate the level range before packing, since an out-of-range level can overwrite bit 15. The pasted IQ1_S decoder was empty, so I checked the [upstream IQ1_S decoder](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py#L1234-L1256).

2. **IQ2_XXS scale bits.** `g[:, :, 7] = (g[:, :, 7] & 15) | (ls.astype(np.uint8) << 4)` matches the supplied decoder. In each eight-byte group, bits 28–31 of the second little-endian `uint32` are the high nibble of byte 7. Its low nibble contains sign-index bits. There are eight levels *fields* per 66-byte block, each selecting one of 16 scales \(d(l+0.5)/4\) for 32 weights. Keep the implementation. The only writable bits are `0xF0` at block-relative bytes \(9+8k\), \(k=0,\ldots,7\).

3. **IQ2_XS scale bits.** `off = 66 if t == Q.IQ2_XS else 74` and `np.stack([s & 15, s >> 4], axis=2)` are correct for XS. The two bytes of `d` and 64 bytes of `qs` put the eight scale bytes at offsets 66–73. Each byte covers 32 weights, with its low nibble scaling the first 16 and its high nibble scaling the next 16. The total is 74 bytes, and the multiplier is \(d(l+0.5)/4\). Keep the offset, order, and inverse packing in `set_ls`.

4. **IQ2_S scale bits.** `Q.IQ2_S: (82, 16, 16, _f_iq2)` and the `off = 74` branch are correct. The decoder consumes two bytes of `d`, 32 index bytes, 32 sign bytes, and eight `qh` bytes before the eight scale bytes at offsets 74–81. The low/high order, 16-weight grouping, and multiplier agree with XS. Keep this branch. The pasted IQ2_S decoder was also empty, so I checked the [upstream IQ2_S decoder](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py#L957-L983).

5. **Input interpretation.** `blocks[:, :2].copy().view(np.float16)` and `np.asarray(ref[n].data).view(np.uint16)` assume compatible byte order, and the latter assumes the reference tensor really is BF16. `reshape` alone cannot establish row/column agreement here, since both dimensions are 2304. Check tensor types, byte order, expert order, moment dimensions, and nine 256-weight blocks per row. Verify a known reference row and its corresponding decoded row through the GGUF reader. Either enforce the little-endian layout throughout or normalize it before using the setters and decoder.

6. **Objective and normal equations.** `return (A * s * s - 2 * Rr * s).sum(-1) + ((P * s).sum(-1) - q) ** 2` is correct for the stated surrogate. Let \(U\) have one column per sub-block, containing its unit codes on that sub-block and zeros elsewhere. For \(V=\operatorname{diag}(v)\), the per-row loss is
   \[
   L(s)=(Us-w)^\top V(Us-w)+[m^\top(Us-w)]^2.
   \]
   Disjoint supports make \(D=U^\top VU=\operatorname{diag}(A)\). With \(r=U^\top Vw,\ p=U^\top m,\ q=m^\top w\), differentiation gives \((D+pp^\top)s=r+pq\). `const` restores \(w^\top Vw\) for reporting. Keep the mean projection across **all nine blocks of the row**. Squaring a separate mean residual for each block would change the objective. A positive expert count multiplies the loss and drops out of its minimizer.

7. **Closed-form solution.** `sig = x - y * ((P * x).sum(-1) / (1 + (P * y).sum(-1)))[:, None]` is Sherman–Morrison when every \(A_j>0\). `Aa = np.where(A > 0, A, 1e-30)` changes the problem at zero curvature and can cause cancellation between enormous terms. \(A_j=0\) can coexist with \(p_j\ne0\) on a constant, nonzero input feature. For positive \(D\), use the equivalent form
   \[
   z=D^{-1}r,\qquad y=D^{-1}p,\qquad
   s=z+y\,\frac{q-p^\top z}{1+p^\top y}.
   \]
   Freeze coordinates with \(A_j=p_j=0\) at their original levels. For other singular rows, starting discrete descent from the original levels avoids both the artificial divisor and the need for a unique continuous solution.

8. **Zero super-block scales.** `safe = np.where(sig0 == 0, 1, sig0)` is harmless for decoded loss while `d` stays fixed. All four level multipliers are strictly positive, so zero `sig0` means zero `d` and every legal level decodes to zero. But `u = deq / safe[..., None]` then recovers zeros and loses the underlying code vector. Preserve the original level bits for these blocks. If refitting `d`, recover their unit codes from the packed fields, for example by decoding a temporary copy with `d=1` and dividing by the known level multipliers.

9. **Discrete rounding and descent.** `ls = np.abs(sig[..., None] - dd[..., None] * lev).argmin(-1)` is a valid initialization. Independent rounding minimizes distance to each continuous coordinate, while the rank-one term couples the choices. `de = ... + (g + P[:, j] * delta) ** 2 - g ** 2` is the correct loss change, and the sequential update to `g` is necessary. But `for _ in range(2)` and `for step in (-1, 1)` allow a coordinate to move at most two levels from its initial value. Use the exact conditional minimum. With \(g=p^\top s-q\) and \(h_j=A_j+p_j^2>0\), it is
   \[
   s_j^*=s_j-\frac{A_js_j-r_j+p_jg}{h_j}.
   \]
   Choose the nearest legal \(d_j f(l)\), retaining the current level on ties. Equivalently, score all 8 or 16 levels using \(\Delta L=2\delta(A_js_j-r_j+p_jg)+h_j\delta^2\), which avoids subtracting two squared residuals. Run from both the original and rounded levels, keep the better row, and sweep until no strict improvement or a stated cap. A completed no-change sweep establishes a one-coordinate optimum. Paired moves can still improve it.

10. **Packed-byte and decode checks.** `changed = np.flatnonzero((new != blocks).any(0))` only logs byte positions, gets overwritten on every expert, and prints at most 12 positions from the last expert. `assert np.allclose(chk, ...)` checks decoded values within tolerance and disappears under `python -O`. Even exact decoded equality can hide changed codes in a zero-`d` block. For **every** block, require `((new ^ blocks) & ~allowed_mask) == 0` bytewise, with a `uint8` mask constructed from points 1–4. For little-endian IQ1_S the allowed byte mask is `0x70` at \(35+2k\). Also require `get_ls(t, new)` to equal the selected levels exactly. Keep the decode check as a separate explicit failure, and reopen the written donor to verify tensor type, logical shape, and payload against the checked bytes.

11. **Keep-if-not-worse.** `keep = e1 > e0  # never worse than llama-quantize` gives a per-row fallback against the supplied `src` under the fitted surrogate, in exact arithmetic with finite data. Row-level fallback is appropriate because all its sub-blocks share a mean residual. The comment overstates the guarantee. Ties rewrite bits, a NaN comparison is false, and `err(...) + const` can lose precision when the true residual is small. Score the original and actual packed candidate decodes directly in float64 using \(e=\hat w-w\) and \(L=\sum_j v_je_j^2+(m^\top e)^2\). Fail on invalid inputs or baseline scores, and accept only a finite improvement exceeding a numerical tolerance. With `cnt[e] == 0`, copy the expert unchanged. As written, zero moments can make every level tie and `argmin` selects level zero. Guard `tot1 / tot0` when the baseline total is zero.

12. **What the input model measures.** `n * [ sum_j v_j dw_j^2 + (m . dw)^2 ]` is a defensible surrogate for a **single linear preactivation** on tokens routed to that expert. Its exact empirical summed squared error is \(n\,\Delta w^\top(\operatorname{Cov}(x)+mm^\top)\Delta w\), using population moments of those tokens. The script keeps the exact mean contribution and exact second-moment diagonal. Its omitted term is \(2n\sum_{i<j}\operatorname{Cov}(x_i,x_j)\Delta w_i\Delta w_j\), of either sign. The reported 20% mean energy and 2.5–5× directional error difference motivate measuring this term of the loss, but neither establishes a routed-output improvement. Validate on held-out routed inputs by reporting the mean contribution and the actual \(\sum_t(\Delta w^\top x_t)^2\) separately. A small routed-token sample also lets you rescore candidate rows with the full empirical covariance without storing a \(2304^2\) matrix.

13. **Router weights and expert mixing.** `tot0 += e0 * cnt[e]` weights by expert visitation. For fixed routing and a router coefficient \(\alpha_{te}\) applied to expert \(e\)'s output, the layer error on token \(t\) is \(\|\sum_e\alpha_{te}\Delta y_{te}\|^2\). An isolated expert contributes with \(\alpha_{te}^2\), and co-selected experts contribute cross terms. For a cheap isolated-expert linear proxy, collect moments with token weight \(\alpha_{te}^2\), normalize them by \(\sum_t\alpha_{te}^2\), and aggregate using that same sum. This requires router-weighted observations when weights vary with the input. For validation, retain the actual top-\(k\) sets and coefficients and compare the summed routed outputs. Use the deployed graph’s actual placement of routing weights when replaying it.

14. **Fused gate/up and SwiGLU.** `new, e0, e1 = refit_expert(ty, blocks, wsrc[e], v, m)` gives every row the same input metric and scores each independently. In the stated layout, gate row \(i\) pairs with up row **\(1152+i\)** for \(0\le i<1152\). For their preactivations \(g,u\), the hidden activation is \(h=\operatorname{SiLU}(g)u\), with first-order error
   \[
   \delta h\approx u\,\operatorname{SiLU}'(g)\,\delta g+
                         \operatorname{SiLU}(g)\,\delta u .
   \]
   Squaring introduces token-dependent sensitivities and a gate/up cross term. The expert down projection \(D_e\) further scores hidden error through \(D_e^\top D_e\), coupling hidden channels. A cheap improvement is to replay a small routed sample and score each gate/up pair by its **exact** change in the SwiGLU product. When changing one pair in isolation, weight its squared hidden error by the squared norm of the corresponding down-projection column. For the final comparison, apply the full down projection and router mix on held-out tokens.

15. **Refitting the fp16 super-block `d`.** `d = blocks[:, :2].copy().view(np.float16)` leaves one shared degree of freedom fixed per 256 weights. With codes and levels fixed inside block \(B\), put \(b_j=f(l_j)u_j\) for its weights, \(P_B=\sum_{j\in B}m_jb_j\), and \(g_{-B}=\sum_{j\notin B}m_j\hat w_j-q\). The exact conditional minimizer under the current surrogate is
   \[
   d_B^*=\frac{\sum_{j\in B}v_jb_jw_j-P_Bg_{-B}}
                   {\sum_{j\in B}v_jb_j^2+P_B^2}.
   \]
   Score the bracketing legal finite fp16 values and the original `d`, keeping the original on ties or zero curvature. Alternate these updates with exact level sweeps. In this variant, extend the byte mask to permit the two `d` bytes. Score the resulting packed decode with the same row-level acceptance check.

16. **Joint code and scale moves.** `u = deq / safe[..., None]` fixes each code shape, so scale fitting alone cannot correct a poorly aligned grid vector or sign pattern. On high-residual sub-blocks, shortlist legal alternative 8-weight codes and their format-specific sign choices, then score each replacement **together with** the best shared 16- or 32-weight level. Include all weights sharing that level and update the row-wide mean residual. This can accept a code/level combination even when either change alone raises loss. Start with the original combination as a candidate, verify the packed decode, and judge the extension by held-out routed-output error. This review verifies the field layouts and algebra; no model tensors or routed-token measurements were available to establish an empirical gain.


## Response

- 1-4 **Accepted as confirmation**: bit fields correct; independently confirmed by decoding with ggml to_float (max difference 0).
- 5, 10, 11 **Accepted**: the keep rule now requires a strict improvement; byte-mask checks noted as hardening.
- 12-14 **Accepted as the explanation of limits**: the surrogate ignores off-diagonal covariance, router weights and SwiGLU coupling. M0r failed on S (+0.07237). The measured mechanism is attenuation (gain 0.79-0.94), recorded in CHANGELOG 07:15.
- 15, 16 **Open**: d refit and joint code moves are not pursued, since the refit line is closed.
