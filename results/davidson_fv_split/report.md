# Davidson FV split: generic heads vs task heads (check 1)

Check: all30 vs the saved FV run at the same alphas: 7/168 rows differ, mean |diff| 0.0005, max 0.03 (the batches differ with the alpha grid, bf16).

## Accuracy, task means, mean over 14 tasks

| set | 0 | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|---|
| all30 | 0.01 | 0.33 | 0.09 | 0 | 0 | 0 |
| task24 | 0.01 | 0.27 | 0.5 | 0.54 | 0.3 | 0.05 |
| drop6_r0 | 0.01 | 0.27 | 0.11 | 0.01 | 0 | 0 |
| drop6_r1 | 0.01 | 0.24 | 0.08 | 0.01 | 0 | 0 |
| drop6_r2 | 0.01 | 0.29 | 0.09 | 0.01 | 0 | 0 |
| generic6 | 0.01 | 0.03 | 0.02 | 0 | 0 | 0 |

## Words in the answer, task means, mean over 14 tasks

| set | 0 | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|---|
| all30 | 9.39 | 3.41 | 5.54 | 4.41 | 2.34 | 4.09 |
| task24 | 9.39 | 7.87 | 4.35 | 3.97 | 7.95 | 6.59 |
| drop6_r0 | 9.39 | 4.01 | 4.69 | 5.58 | 2.62 | 4.59 |
| drop6_r1 | 9.39 | 4.34 | 5.45 | 5.14 | 2.66 | 4.46 |
| drop6_r2 | 9.39 | 3.41 | 4.7 | 5.42 | 2.34 | 3.73 |
| generic6 | 9.39 | 8.26 | 7.44 | 7.54 | 2.92 | 4.2 |

## Share stopping at a line break, task means, mean over 14 tasks

| set | 0 | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|---|
| all30 | 0.74 | 0.92 | 0.8 | 0.78 | 0.88 | 0.84 |
| task24 | 0.74 | 0.75 | 0.86 | 0.88 | 0.76 | 0.81 |
| drop6_r0 | 0.74 | 0.9 | 0.83 | 0.71 | 0.87 | 0.8 |
| drop6_r1 | 0.74 | 0.88 | 0.81 | 0.73 | 0.87 | 0.81 |
| drop6_r2 | 0.74 | 0.92 | 0.84 | 0.72 | 0.88 | 0.84 |
| generic6 | 0.74 | 0.74 | 0.72 | 0.62 | 0.88 | 0.82 |

## Accuracy with the control mean

| set | 0 | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|---|
| all30 | 0.01 | 0.03 | 0.03 | 0.01 | 0 | 0 |
| generic6 | 0.01 | 0.02 | 0.03 | 0.01 | 0 | 0 |

## Best alpha per task (on the same items)

| task | all30 | task24 | drop6_r0 | drop6_r1 | drop6_r2 | generic6 | alpha all30 | alpha task24 |
|---|---|---|---|---|---|---|---|---|
| antonym | 0.26 | 0.61 | 0.25 | 0.22 | 0.14 | 0.04 | 1 | 3 |
| country-capital | 0.83 | 1 | 0.34 | 0.31 | 0.79 | 0.02 | 1 | 2 |
| country-currency | 0.13 | 0.64 | 0.17 | 0.07 | 0.15 | 0 | 1 | 3 |
| english-french | 0.09 | 0.5 | 0.09 | 0.08 | 0.08 | 0.05 | 1 | 3 |
| english-german | 0.02 | 0.39 | 0.01 | 0 | 0.01 | 0 | 1 | 3 |
| english-spanish | 0.1 | 0.7 | 0.11 | 0.01 | 0.01 | 0.01 | 1 | 3 |
| landmark-country | 0.89 | 0.95 | 0.67 | 0.51 | 0.84 | 0.01 | 1 | 2 |
| next_item | 0.11 | 0.3 | 0.11 | 0.14 | 0.12 | 0.08 | 2 | 8 |
| park-country | 0.89 | 0.91 | 0.81 | 0.79 | 0.88 | 0.1 | 1 | 2 |
| present-past | 0.02 | 0.04 | 0.03 | 0.01 | 0.01 | 0 | 1 | 3 |
| prev_item | 0.01 | 0.04 | 0.01 | 0.01 | 0.01 | 0.01 | 1 | 8 |
| product-company | 0.63 | 0.82 | 0.62 | 0.59 | 0.62 | 0.11 | 1 | 2 |
| singular-plural | 0.59 | 0.92 | 0.57 | 0.69 | 0.5 | 0.03 | 1 | 2 |
| synonym | 0.04 | 0.35 | 0.04 | 0.04 | 0.04 | 0.04 | 0 | 3 |

Mean: all30 0.33, task24 0.58, drop6_r0 0.27, drop6_r1 0.25, drop6_r2 0.30, generic6 0.04

## Most common answers (task means)

- all30, alpha 1: `Apple` 15, `Germany` 14, `Australia` 10, `Argentina` 10, `India` 9
- all30, alpha 3: `://` 442, `1` 72, `. 1. 2. 3. 4. 5. 6. ` 57, `.swing` 55, `://www.bbc.com/news/` 28
- generic6, alpha 5: `.swing` 445, `://` 305, `.swinging` 35, `.swinging into the 2` 24, `://www.abc.net.au/ne` 19
- task24, alpha 3: `India` 16, `Apple` 16, `Australia` 15, `Germany` 14, `1. 2. 3. 4. 5. 6. 7.` 13

## FV part norms

| task | all30 | generic6 | task24 | cos(generic6, task24) |
|---|---|---|---|---|
| antonym | 14.2 | 12.8 | 5.3 | 0.08 |
| country-capital | 13.0 | 10.6 | 7.6 | -0.00 |
| country-currency | 12.1 | 9.9 | 7.1 | -0.03 |
| english-french | 11.8 | 10.7 | 4.4 | 0.06 |
| english-german | 10.7 | 9.4 | 4.8 | 0.03 |
| english-spanish | 11.6 | 10.3 | 5.0 | 0.05 |
| landmark-country | 14.4 | 12.7 | 6.1 | 0.05 |
| next_item | 15.0 | 13.8 | 4.6 | 0.11 |
| park-country | 14.6 | 12.8 | 6.5 | 0.05 |
| present-past | 14.4 | 12.1 | 7.0 | 0.06 |
| prev_item | 16.5 | 15.3 | 4.9 | 0.09 |
| product-company | 12.3 | 10.2 | 6.4 | 0.04 |
| singular-plural | 14.5 | 12.4 | 7.0 | 0.05 |
| synonym | 12.7 | 11.6 | 4.8 | 0.05 |

Mean cosine across task pairs: generic6 0.96, task24 0.34.
