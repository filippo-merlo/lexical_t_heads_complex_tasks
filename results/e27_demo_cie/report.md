# E27: causal heads from instructions vs demonstrations (Davidson et al. Sec. 3.3)

Davidson et al., Llama-3.1-8B-Instruct: 4 of the top 20 shared.

## 10-shot accuracy (first token; prompts tried until 125 correct)

|  | all | A | B |
|---|---|---|---|
| antonym | 0.68 | 0.728 | 0.675 |
| synonym | 0.488 | 0.493 | 0.512 |
| singular-plural | 0.981 | 1.0 | 0.971 |
| present-past | 0.979 | 0.993 | 0.986 |
| country-capital | 0.948 | 0.99 | 0.959 |
| country-currency | 0.691 | 0.707 | 0.776 |
| landmark-country | 0.893 | 0.907 | 0.893 |
| park-country | 0.907 | 0.95 | 0.875 |
| product-company | 0.812 | 0.888 | 0.85 |
| next_item | 0.896 | 0.929 | 0.938 |
| prev_item | 0.88 | 0.867 | 0.893 |
| english-french | 0.812 | 0.812 | 0.844 |
| english-german | 0.571 | 0.591 | 0.554 |
| english-spanish | 0.881 | 0.844 | 0.838 |

## One set over the 14 tasks: heads in both top-N

| n | A | B | all | chance | chance_layer |
|---|---|---|---|---|---|
| 20 | 3 | 3 | 3 | 0.4 | 1.7 |
| 40 | 11 | 13 | 12 | 1.6 | 6.1 |
| 60 | 18 | 19 | 19 | 3.5 | 10.7 |
| 80 | 26 | 29 | 31 | 6.2 | 15.7 |
| 100 | 32 | 34 | 42 | 9.8 | 21.9 |
| 120 | 40 | 46 | 51 | 14.1 | 28.5 |

## Per task: heads in both top-N, mean over the 14 tasks

| split | n | shared | chance | chance_layer |
|---|---|---|---|---|
| A | 20 | 2.5 | 0.39 | 1.16 |
| A | 40 | 8.0 | 1.56 | 4.21 |
| A | 60 | 14.07 | 3.52 | 8.33 |
| A | 80 | 20.93 | 6.25 | 13.08 |
| A | 100 | 27.21 | 9.77 | 17.88 |
| A | 120 | 34.0 | 14.06 | 22.85 |
| B | 20 | 2.43 | 0.39 | 1.13 |
| B | 40 | 8.21 | 1.56 | 4.05 |
| B | 60 | 15.0 | 3.52 | 7.83 |
| B | 80 | 21.29 | 6.25 | 12.2 |
| B | 100 | 26.86 | 9.77 | 17.06 |
| B | 120 | 32.79 | 14.06 | 22.1 |
| all | 20 | 2.93 | 0.39 | 1.17 |
| all | 40 | 8.21 | 1.56 | 4.33 |
| all | 60 | 14.43 | 3.52 | 8.39 |
| all | 80 | 20.93 | 6.25 | 13.29 |
| all | 100 | 28.0 | 9.77 | 18.39 |
| all | 120 | 35.21 | 14.06 | 24.15 |

## Rank correlation of the two scores over the 1024 heads

| split | scope | rank_corr |
|---|---|---|
| all | pooled | 0.209 |
| A | pooled | 0.124 |
| B | pooled | 0.164 |

Per task (mean, min, max): A: 0.06 (-0.06, 0.20); B: 0.09 (-0.02, 0.16); all: 0.09 (0.03, 0.17)

## Mean layer of the top 20 (one set over the tasks)

| split | scope | instruction_only | demonstration_only | shared | instruction_all | demonstration_all |
|---|---|---|---|---|---|---|
| all | pooled | 15.6 | 13.2 | 20.3 | 16.3 | 14.3 |
| A | pooled | 14.9 | 13.8 | 15.3 | 15.0 | 14.0 |
| B | pooled | 15.0 | 13.1 | 21.3 | 16.0 | 14.3 |

Check: the held-out instruction rankings reproduce head-sets-simple's Causal sets (336 sets).
