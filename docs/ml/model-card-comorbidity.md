# Model Card: Charlson Comorbidity Index (CCI) Calculator

## Overview

The Charlson Comorbidity Index calculator quantifies disease burden by
mapping ICD-10-CM codes and/or free-text clinical narratives to the 17
standard CCI disease categories. It produces a composite score predicting
10-year mortality risk, with optional age adjustment per the Charlson–Deyo
adaptation.

## Architecture

```
Input (ICD-10-CM codes + free text)
  │
  ├── Phase 1: Prefix-based ICD-10 matching (O(n×p) worst-case)
  │     └── ~200 validated code prefixes across 17 categories
  │
  ├── Phase 2: Regex-based text extraction (17 pattern groups)
  │     └── ICD-coded conditions take priority over text matches
  │
  ├── Phase 3: Hierarchical exclusion (3 pairs)
  │     └── Mild variant removed when severe variant present
  │
  ├── Phase 4: Score computation (sum of category weights)
  │
  ├── Phase 5: Age adjustment (+1 per decade above 50, max 4)
  │
  └── Phase 6: Mortality estimation (Charlson exponential formula)
        └── 1 − 0.983^(e^(CCI × 0.9))
```

## Disease Categories

| # | Category | Weight | ICD-10 Prefixes |
|---|----------|--------|-----------------|
| 1 | Myocardial Infarction | 1 | I21, I22, I252 |
| 2 | Congestive Heart Failure | 1 | I099, I110, I130, I50, ... |
| 3 | Peripheral Vascular Disease | 1 | I70, I71, I731, ... |
| 4 | Cerebrovascular Disease | 1 | G45, G46, I60–I69 |
| 5 | Dementia | 1 | F00–F03, G30, G311 |
| 6 | Chronic Pulmonary Disease | 1 | J40–J47, J60–J67, ... |
| 7 | Rheumatic Disease | 1 | M05, M06, M32–M34, ... |
| 8 | Peptic Ulcer Disease | 1 | K25–K28 |
| 9 | Mild Liver Disease | 1 | B18, K700–K709, K73, K74, ... |
| 10 | Diabetes (Uncomplicated) | 1 | E100–E149 (select) |
| 11 | Diabetes (Complicated) | 2 | E102–E147 (select) |
| 12 | Hemiplegia/Paraplegia | 2 | G81, G82, G830–G839, ... |
| 13 | Renal Disease | 2 | N18, N19, I120, Z992, ... |
| 14 | Malignancy | 2 | C00–C76, C81–C97 |
| 15 | Moderate/Severe Liver Disease | 3 | I850, K704, K721, ... |
| 16 | Metastatic Solid Tumor | 6 | C77–C80 |
| 17 | AIDS/HIV | 6 | B20–B24 |

## Hierarchical Exclusion Rules

When both mild and severe forms are detected, only the severe form counts:

1. **Diabetes**: Uncomplicated (1) excluded when Complicated (2) present
2. **Liver**: Mild (1) excluded when Moderate/Severe (3) present
3. **Cancer**: Malignancy (2) excluded when Metastatic (6) present

## Risk Groups

| CCI Score | Risk Group | 10-Year Mortality (approx.) |
|-----------|-----------|----------------------------|
| 0 | Low | <2% |
| 1–2 | Mild | 2–26% |
| 3–4 | Moderate | 26–52% |
| ≥5 | Severe | >52% |

## Text Extraction

Each category has compiled regex patterns matching common clinical
phrases (e.g., "myocardial infarction", "STEMI", "CHF"). Confidence
is based on match specificity:

- Multi-word terms (≥3 words): 0.85
- Two-word terms: 0.80
- Long abbreviations (≥4 chars): 0.75
- Short abbreviations (<4 chars): 0.70

## Performance

- **ICD matching**: <0.1ms per code (prefix lookup)
- **Text extraction**: <1ms per document (compiled regex)
- **Total calculation**: <2ms per patient

## API Endpoints

- `POST /comorbidity` — Single patient CCI
- `POST /comorbidity/batch` — Up to 50 patients
- `GET /comorbidity/categories` — List all 17 categories
- `GET /comorbidity/categories/{name}` — Category detail with code prefixes

## Limitations

1. **ICD-10-CM only** — Does not support ICD-9-CM or SNOMED codes
2. **Text extraction is supplementary** — May miss conditions expressed
   in unusual phrasing or miss context-dependent negation
3. **No temporal reasoning** — Cannot distinguish current from resolved
   conditions (e.g., cancer in remission still matches)
4. **Mortality estimates are population-level** — Individual patient
   outcomes depend on many factors not captured by CCI alone

## References

1. Charlson ME et al. (1987). J Chronic Dis. 40(5):373-383.
2. Quan H et al. (2005). Med Care. 43(11):1130-1139.
3. Deyo RA et al. (1992). J Clin Epidemiol. 45(6):613-619.
