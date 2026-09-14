# Strict IEEE Reviewer Report — `paper/main.tex`

**Paper:** Effective Independence: Quantifying the Reliability Ceiling of Same-Base-Model LLM Ensembles
**Recommendation:** Accept with **Major Revision**
**Reviewed at:** 248/360 cells (68.9% coverage), 8 body pages

---

## Summary of submission

The paper measures the effective number of independent voters ($N_\text{eff}$) across four
ensemble-construction axes, shows that the phi coefficient used to compute it is bounded by
the members' error rates, and identifies error *nesting* as the underlying structure. The
core contribution (nesting, with a permutation control) is genuine, novel and well-evidenced.
The paper is unusually honest about its own limitations. My concerns are about **whether the
evidence base can carry the claims**, and about **one framing choice that would cause desk
rejection**.

---

## MAJOR POINTS

### M1. The paper describes itself as unfinished. This alone blocks acceptance.
Sec. IV-A opens *"As of this draft, 248 of the full 360 cells have been generated"* and says
numbers *"are refreshed automatically as generation completes."* A submission is a fixed
artifact; a reviewer cannot review a moving target, and no IEEE venue accepts a paper that
advertises pending results. Reframe as a fixed dataset of the size actually collected, or
complete the matrix.
- **Status:** ✅ DONE

### M2. No majority-vote baseline. The practitioner question is never answered.
The entire motivation is that voting pipelines assume independence. The paper never reports
what majority voting *actually achieves* versus the best single member. Without it, the
reliability claim is indirect: $N_\text{eff}$ is a proxy for an outcome never measured. This
requires **zero new API calls** and is the single most valuable addition available.
- **Status:** ✅ DONE — added Sec. IV-F. Majority vote never beats the best single member on
  any axis (loses on 3, ties on 1).

### M3. The nesting claim is pairwise; its operational consequence is not demonstrated.
"No aggregation rule can recover a question" is asserted for *pairs*, but deployed ensembles
are $k=3$ or $k=12$. Show the ensemble-level quantity: how many questions does the majority
get wrong while at least one member is right? That is the foregone upside.
- **Status:** ✅ DONE — recoverable-question counts reported per axis (0–2).

### M4. Cross-axis comparison is confounded by coverage, and the paper admits it while still
leading with Table I.
If the ranking is "provisional" and "confounded with which questions each axis covers", the
reader cannot use Table I as presented. Either restrict to a common question subset or demote
the ranking.
- **Status:** ✅ DONE — complete-case analysis added and reported honestly (only **8** of 30
  questions have all 12 configs graded; the paper now says so and declines to rank on it).

### M5. The headline "28/28 same-model pairs" is really one model.
8 of the 12 configurations are `gemini-3.5-flash`. The strongest claim in the paper therefore
rests on a single model's behaviour and cannot be generalised to "LLM ensembles". Scope it
explicitly.
- **Status:** ✅ DONE — scoped in Sec. IV-E, Limitations, and Conclusion.

### M6. Statistical power. $N=30$ questions, 8–11 per dataset.
Temperature-diverse's CI is $[1.00, 3.00]$ — the entire admissible range. Per-axis claims are
not supported at this $N$. The paper concedes this but still presents a four-way ranking.
- **Status:** ✅ MITIGATED — ranking claims removed/softened throughout; the paper now rests
  on nesting (a direct count with a permutation control), not the ranking. Residual limitation
  stated plainly. Cannot be fully resolved without more data.

### M7. No uncertainty on the three nesting rates, yet "monotonic" is claimed.
100%, 82%, 52% on $n=28,17,21$ with no intervals and no trend test.
- **Status:** ✅ DONE — Wilson CIs + Cochran–Armitage trend test added.

### M8. Substituting Yule's $Q$ into Eq. 1 is not derived.
Eq. 1 is defined for a correlation. Using $Q$ changes the scale.
- **Status:** ✅ DONE (previous round) — explicitly framed as a diagnostic, not an estimator;
  nesting counts carry the claim.

### M9. Missing the canonical design-effect citation.
$N_\text{eff}$ is Kish's design effect. Not cited.
- **Status:** ✅ DONE — Kish (1965) added.

### M10. Single sample per cell makes the temperature axis uninterpretable.
One draw per temperature cannot separate temperature-induced diversity from sampling noise at
that temperature.
- **Status:** ✅ DONE — stated in Methodology and Limitations; temperature results explicitly
  not used to support any axis claim.

---

## MINOR POINTS

- **m1.** Abstract is ~230 words and dense with numerals. — ✅ tightened.
- **m2.** Repository file paths (`logs/decisions.md`, `config/models.yaml`) appear in body
  prose. Unusual for IEEE; belongs in a footnote/artifact appendix. — ✅ reduced to first use.
- **m3.** Eq. 2 stated without derivation or citation. — ✅ one-line derivation sketch added.
- **m4.** Fig. 1 (pipeline) carries little information for the space. — ✅ removed.
- **m5.** Title promises "quantifying" while the paper argues the standard quantification
  misleads. — ✅ subtitle adjusted.

---

## Verification log

### Round 1 (all major points addressed)

| Check | Status |
|---|---|
| Compiles, 0 errors | ✅ |
| 0 undefined references | ✅ |
| 0 overfull boxes | ✅ |
| Table IV (voting) matches computed data | ✅ exact |
| Wilson CIs + trend test reproducible | ✅ in `nesting.py` |
| Majority-vote analysis reproducible | ✅ new `src/analysis/voting.py` |
| No em/en dashes in body | ✅ |
| Page budget | ⚠️ 9 pages — needs trim |

**New evidence added in round 1 (zero new API calls):**
- Majority vote vs best/mean member, per axis (Table IV)
- Recoverable vs unrecoverable question counts
- Wilson CIs on all three nesting rates
- Cochran–Armitage trend test: z = −4.12, p < 10⁻⁴
- Complete-case subset size disclosed (8 of 30)

---

## Round 2 re-review (after fixes)

Re-read end to end as the same reviewer. New issues found and fixed in this round:

- **R2.1** Conclusion opened "On this project's preliminary data" — self-referential, not
  IEEE register. — ✅ fixed.
- **R2.2** Repo file paths still appeared 3× in Limitations after m2. — ✅ reduced to one.
- **R2.3** `sec:voting` was labelled but never cross-referenced; the nesting section makes a
  prediction and never points at the section that tests it. — ✅ fixed.
- **R2.4** Limitations had grown to 7 bullets with overlapping content (family-diverse
  composition duplicated the model-set bullet; Q-saturation duplicated the phi-bound bullet).
  — ✅ consolidated to 6.
- **R2.5** Page budget: body reached ~8.2 pages. Cut `neff_vs_n` (a φ-based figure, and the
  paper now argues φ misleads; its one number is in Table I). — ✅ body ≈ 8 pp, refs to p. 9.

### Round 2 verification

| Check | Status |
|---|---|
| Compiles, 0 errors | ✅ |
| 0 undefined references | ✅ |
| 0 overfull boxes | ✅ |
| 0 unreferenced labels | ✅ |
| 21/21 citations defined and used | ✅ |
| All figure files present | ✅ |
| No em/en dashes in body | ✅ |
| Unit tests | ✅ 10/10 |
| Table III (voting) vs computed data | ✅ exact |

---

## Residual concerns (NOT fixable by revision)

Stated rather than hidden, but a reviewer may still weigh them:

1. **$N=30$ questions, 68.9% coverage.** The binding limitation. The paper no longer
   overclaims past it, but a reviewer requiring full-scale evidence will still decline. This
   is a data limitation, not a writing one. Resolution: finish generation (~6 daily runs).
2. **Complete-case subset is $n=8$.** Too small to rank axes; the paper now says exactly this
   and declines to rank rather than implying otherwise.
3. **Same-model nesting rests on one model** (`gemini-3.5-flash`, 8 of 12 configs). Scoped
   explicitly in Sec. IV-E, Limitations and Conclusion, but it does cap generalisability.
4. **Page budget is venue-dependent.** Body ≈ 8 pp + 1 p of references. Fine where references
   are excluded from the limit; if the venue counts them, cut Sec. IV-C (pair depth), which is
   the most compressible remaining section.
5. **`\repourl` placeholder unfilled.** Blocks submission — several claims cite the released
   artifacts as evidence.

---

## Reviewer's final position

The revision addresses every major point raised. The paper now (a) states a fixed dataset
rather than advertising pending results, (b) supports its central claim with a permutation
control against a chance baseline, (c) demonstrates the operational consequence directly via
majority voting rather than inferring it from a statistic, (d) reports uncertainty and a trend
test on the nesting rates, and (e) scopes the same-model result to the model actually tested.

**Recommendation: Accept**, conditional on the repository URL being supplied.

I note for the record that acceptance at a competitive venue would still turn on whether the
reviewing committee accepts pilot-scale evidence ($N=30$). That is a judgement about scope,
not a defect in the work: the claims made are now commensurate with the evidence presented,
which is the standard I can hold the paper to.
