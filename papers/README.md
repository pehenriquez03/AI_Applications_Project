# Research papers for the factor-extraction evaluation set

This folder holds the real research papers the LLM extraction step is scored
against, plus the hand-written answer key.

- `pdf/` — the papers. Each file name encodes author, year, topic, and the
  version downloaded (NBER working paper, journal version, or author copy).
- `answer_key.json` — what a correct extraction should contain for each paper.
  **Never put this file or its contents in a prompt.** It is the grading key.
- `MANIFEST.sha256` — checksums of the PDFs as downloaded on 2026-10-27.

## Status of the set

| Role | Paper | File | Notes |
|---|---|---|---|
| core | Jegadeesh & Titman 2001, momentum | `jegadeesh_titman_2001_momentum_nber7159.pdf` | NBER WP; substitutes for JT 1993 (no free copy) |
| core | Lehmann 1990, weekly reversal | `lehmann_1990_fads_martingales_nber2533.pdf` | NBER WP; substitutes for Jegadeesh 1990 |
| core | George & Hwang 2004, 52-week high | `george_hwang_2004_52week_high_jf.pdf` | published, author copy |
| core | Ang, Hodrick, Xing, Zhang 2006, idiosyncratic volatility | `ang_hodrick_xing_zhang_2006_volatility_nber10852.pdf` | NBER WP |
| core | Frazzini & Pedersen 2014, betting against beta | `frazzini_pedersen_2014_betting_against_beta_jfe.pdf` | published, CC BY-NC-ND |
| core | Bali, Cakici, Whitelaw 2011, MAX | `bali_cakici_whitelaw_2011_maxing_out_nber14804.pdf` | NBER WP |
| core (missing) | Moskowitz & Grinblatt 1999, industry momentum | — | no free full text; get from the library via DOI 10.1111/0022-1082.00146 |
| core | Amihud 2002, illiquidity | `amihud_2002_illiquidity_jfm.pdf` | published; needs daily volume (not in repo) |
| core | Heston & Sadka 2008, seasonality | `heston_sadka_2008_seasonality_wp.pdf` | 2006 working paper |
| core-hard | Da, Gurun, Warachka 2014, frog in the pan | `da_gurun_warachka_2014_frog_in_the_pan_rfs.pdf` | published, author copy |
| optional (missing) | Blitz & van Vliet 2007, volatility effect | — | SSRN, free login required |
| decoy | Fama & French 1993 | `decoy_fama_french_1993_common_risk_factors_jfe.pdf` | should return cannot-implement (multifactor, needs book equity) |
| decoy | Novy-Marx 2013, gross profitability | `decoy_novy_marx_2013_gross_profitability_nber15940.pdf` | should return cannot-implement (needs fundamentals) |
| decoy | Moskowitz, Ooi, Pedersen 2012, time-series momentum | `decoy_moskowitz_ooi_pedersen_2012_time_series_momentum_jfe.pdf` | should return cannot-implement (futures, not equities) |
| decoy | Asness, Moskowitz, Pedersen 2013, value and momentum everywhere | `decoy_asness_moskowitz_pedersen_2013_value_momentum_everywhere_jf.pdf` | should return cannot-implement or flagged partial |

## How the answer key was drafted

Field values were taken from the papers' own text (abstract, methodology
section, and the headline table) using text extraction, then recorded with a
`confidence` tag. `high` means the number appears verbatim in the paper text.
`verify` means the field was drafted from a summary or a nearby passage and a
group member should open the PDF and confirm the table value before grading
with it. Two entries (Lehmann, Moskowitz-Grinblatt) have a `null` headline
value for that reason.

## Licensing

NBER working papers, the CBS open-access copy, and author-hosted copies are
free to download for research and teaching. The UPenn, Houston, and Yale
copies are course-reading mirrors of published articles; keep them inside the
project for coursework and do not redistribute. Published versions behind
Wiley or Elsevier paywalls are available through the Columbia library.
