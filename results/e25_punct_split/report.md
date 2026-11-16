# E25: punctuation (P) and stopwords (W) as separate filters, 40 heads, check 1

S = attention to punctuation + NLTK stopwords (Bai et al.'s stopword type); P = punctuation only; W = stopwords only. In the three leading S combinations S is replaced by P and by W. Held-out items, cross-fitted alpha in {1, 2}, mean of the two folds and of the 14 tasks. S rows: E22 (same protocol). Shared heads: share of the 40 heads also in the S version, mean over tasks and folds.

## Mean over tasks

|  | task mean | control | fold 1 | fold 2 | shared heads with S version |
|---|---|---|---|---|---|
| ('per task', 'DBTMS40') | 0.61 | 0.023 | 0.608 | 0.611 | 1.0 |
| ('per task', 'DBTMP40') | 0.592 | 0.025 | 0.603 | 0.582 | 0.917 |
| ('per task', 'DBTMW40') | 0.606 | 0.018 | 0.6 | 0.611 | 0.92 |
| ('shared', 'shared_DBTMS40') | 0.533 | 0.018 | 0.515 | 0.551 | 1.0 |
| ('shared', 'shared_DBTMP40') | 0.549 | 0.019 | 0.556 | 0.541 | 0.975 |
| ('shared', 'shared_DBTMW40') | 0.483 | 0.019 | 0.479 | 0.487 | 0.938 |
| ('shared', 'shared_DBTLS40') | 0.603 | 0.022 | 0.598 | 0.608 | 1.0 |
| ('shared', 'shared_DBTLP40') | 0.56 | 0.018 | 0.571 | 0.549 | 0.888 |
| ('shared', 'shared_DBTLW40') | 0.556 | 0.021 | 0.546 | 0.566 | 0.938 |

## By task type: per task DBTMS

| group | DBTMS40 | DBTMP40 | DBTMW40 |
|---|---|---|---|
| Word relations | 0.58 | 0.58 | 0.54 |
| Morphology | 0.69 | 0.57 | 0.63 |
| Facts | 0.83 | 0.8 | 0.82 |
| Sequences | 0.11 | 0.15 | 0.15 |
| Translations | 0.55 | 0.56 | 0.58 |

## By task type: shared DBTMS

| group | shared_DBTMS40 | shared_DBTMP40 | shared_DBTMW40 |
|---|---|---|---|
| Word relations | 0.45 | 0.49 | 0.41 |
| Morphology | 0.77 | 0.8 | 0.67 |
| Facts | 0.71 | 0.7 | 0.7 |
| Sequences | 0.13 | 0.17 | 0.1 |
| Translations | 0.4 | 0.42 | 0.3 |

## By task type: shared DBTLS

| group | shared_DBTLS40 | shared_DBTLP40 | shared_DBTLW40 |
|---|---|---|---|
| Word relations | 0.42 | 0.37 | 0.38 |
| Morphology | 0.87 | 0.81 | 0.72 |
| Facts | 0.87 | 0.84 | 0.87 |
| Sequences | 0.24 | 0.16 | 0.21 |
| Translations | 0.34 | 0.31 | 0.28 |

## Per task

| task | DBTMS40 | DBTMP40 | DBTMW40 | shared_DBTMS40 | shared_DBTMP40 | shared_DBTMW40 | shared_DBTLS40 | shared_DBTLP40 | shared_DBTLW40 |
|---|---|---|---|---|---|---|---|---|---|
| antonym | 0.54 | 0.6 | 0.46 | 0.52 | 0.58 | 0.43 | 0.46 | 0.48 | 0.43 |
| country-capital | 0.97 | 0.94 | 0.97 | 0.85 | 0.83 | 0.8 | 0.98 | 0.96 | 0.98 |
| country-currency | 0.67 | 0.62 | 0.64 | 0.59 | 0.61 | 0.56 | 0.7 | 0.66 | 0.66 |
| english-french | 0.45 | 0.48 | 0.57 | 0.33 | 0.36 | 0.25 | 0.32 | 0.28 | 0.24 |
| english-german | 0.47 | 0.48 | 0.43 | 0.21 | 0.23 | 0.14 | 0.17 | 0.14 | 0.12 |
| english-spanish | 0.71 | 0.72 | 0.74 | 0.65 | 0.66 | 0.5 | 0.54 | 0.53 | 0.47 |
| landmark-country | 0.8 | 0.78 | 0.84 | 0.65 | 0.66 | 0.66 | 0.95 | 0.92 | 0.95 |
| next_item | 0.08 | 0.11 | 0.06 | 0.09 | 0.12 | 0.06 | 0.17 | 0.11 | 0.15 |
| park-country | 0.85 | 0.83 | 0.78 | 0.72 | 0.68 | 0.75 | 0.88 | 0.82 | 0.89 |
| present-past | 0.46 | 0.23 | 0.48 | 0.63 | 0.69 | 0.57 | 0.86 | 0.76 | 0.55 |
| prev_item | 0.14 | 0.19 | 0.23 | 0.17 | 0.23 | 0.15 | 0.31 | 0.21 | 0.27 |
| product-company | 0.86 | 0.82 | 0.86 | 0.75 | 0.72 | 0.74 | 0.86 | 0.85 | 0.87 |
| singular-plural | 0.91 | 0.91 | 0.78 | 0.9 | 0.91 | 0.77 | 0.89 | 0.86 | 0.88 |
| synonym | 0.62 | 0.56 | 0.63 | 0.39 | 0.41 | 0.38 | 0.37 | 0.26 | 0.33 |
