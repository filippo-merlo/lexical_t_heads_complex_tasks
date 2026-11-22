# Signed screen: Selected with features subtracted (40 heads, TF-IDF lists)

Selected: per task DBTSPL, shared DBTM (both folds). A subtracted feature ranks heads by it from the bottom. Quick pass: alpha 2, 10 test items per task, first-token accuracy, task means only; mean over template means, then over the 14 tasks.

Check: Selected's quick rows equal the stored TF-IDF quick pass to 0.0000 (every task, template mean).

## Quick pass, per task

| combination | heads shared with Selected (f1/f2) | fold 1 | fold 2 | mean | minus Selected |
|---|---|---|---|---|---|
| DBTSPL-Q | 29.8/29.4 | 0.659 | 0.590 | 0.624 | +0.014 |
| DBTSPL-OQ | 24.1/24.1 | 0.647 | 0.576 | 0.611 | +0.001 |
| DBTSPL (Selected) | 40/40 | 0.650 | 0.571 | 0.611 | +0.000 |
| DBTSPL-MQ | 27.1/27 | 0.637 | 0.581 | 0.609 | -0.001 |
| DBTSPL-O | 31/30.7 | 0.664 | 0.554 | 0.609 | -0.001 |
| DBTSPL-OMQ | 22.7/22.4 | 0.623 | 0.554 | 0.589 | -0.022 |
| DBTSPL-M | 32.5/32.6 | 0.539 | 0.457 | 0.498 | -0.113 |
| DBTSPL-OM | 27.5/27.2 | 0.491 | 0.431 | 0.461 | -0.149 |

Shortlist (beats Selected on the other fold's quick pass, at most 3): fold 1: DBTSPL-Q, DBTSPL-MQ, DBTSPL-OQ; fold 2: DBTSPL-O, DBTSPL-Q

## Quick pass, shared

| combination | heads shared with Selected (f1/f2) | fold 1 | fold 2 | mean | minus Selected |
|---|---|---|---|---|---|
| DBTM-Q | 24/25 | 0.661 | 0.601 | 0.631 | +0.045 |
| DBTM (Selected) | 40/40 | 0.616 | 0.557 | 0.586 | +0.000 |
| DBTM-O | 28/27 | 0.619 | 0.554 | 0.586 | +0.000 |
| DBTM-PQ | 24/22 | 0.616 | 0.547 | 0.581 | -0.005 |
| DBTM-OQ | 19/19 | 0.613 | 0.539 | 0.576 | -0.011 |
| DBTM-SQ | 22/23 | 0.596 | 0.551 | 0.574 | -0.013 |
| DBTM-S | 30/28 | 0.560 | 0.514 | 0.537 | -0.049 |
| DBTM-P | 31/30 | 0.541 | 0.493 | 0.517 | -0.069 |
| DBTM-OP | 13/12 | 0.516 | 0.467 | 0.491 | -0.095 |
| DBTM-OQL | 15/16 | 0.494 | 0.399 | 0.446 | -0.140 |
| DBTM-SPQ | 9/9 | 0.473 | 0.409 | 0.441 | -0.146 |
| DBTM-SP | 14/13 | 0.437 | 0.401 | 0.419 | -0.167 |
| DBTM-SQL | 13/14 | 0.397 | 0.357 | 0.377 | -0.209 |
| DBTM-OS | 12/12 | 0.341 | 0.323 | 0.332 | -0.254 |
| DBTM-L | 24/25 | 0.316 | 0.310 | 0.313 | -0.274 |
| DBTM-OSQ | 7/6 | 0.321 | 0.304 | 0.313 | -0.274 |
| DBTM-OPQ | 6/6 | 0.321 | 0.281 | 0.301 | -0.285 |
| DBTM-SL | 17/18 | 0.317 | 0.279 | 0.298 | -0.289 |
| DBTM-QL | 17/18 | 0.294 | 0.286 | 0.290 | -0.296 |
| DBTM-OSP | 5/4 | 0.249 | 0.296 | 0.272 | -0.314 |
| DBTM-OSL | 11/11 | 0.271 | 0.267 | 0.269 | -0.317 |
| DBTM-PQL | 14/15 | 0.294 | 0.189 | 0.241 | -0.345 |
| DBTM-OPL | 11/10 | 0.276 | 0.184 | 0.230 | -0.356 |
| DBTM-OSQL | 3/3 | 0.197 | 0.203 | 0.200 | -0.386 |
| DBTM-PL | 15/14 | 0.209 | 0.154 | 0.181 | -0.405 |
| DBTM-OL | 21/21 | 0.203 | 0.150 | 0.176 | -0.410 |
| DBTM-OPQL | 1/1 | 0.069 | 0.076 | 0.072 | -0.514 |
| DBTM-OSPL | 1/1 | 0.063 | 0.077 | 0.070 | -0.516 |
| DBTM-SPL | 10/11 | 0.061 | 0.073 | 0.067 | -0.519 |
| DBTM-OSPQ | 3/3 | 0.051 | 0.034 | 0.043 | -0.544 |
| DBTM-OSPQL | 1/1 | 0.021 | 0.041 | 0.031 | -0.555 |
| DBTM-SPQL | 2/2 | 0.016 | 0.030 | 0.023 | -0.564 |

Shortlist (beats Selected on the other fold's quick pass, at most 3): fold 1: DBTM-Q; fold 2: DBTM-Q, DBTM-O

## Full pass (alphas 1 and 2, all test items, task means only)

Check: Selected's full rows equal the stored TF-IDF full pass to 0.0000.

### per task

| combination | fold 1 | fold 2 |
|---|---|---|
| DBTSPL (Selected) | 0.643 | 0.633 |
| DBTSPL-O | 0.640 | 0.599 |
| DBTSPL-Q | 0.658 | 0.648 |
| DBTSPL-OQ | 0.625 | 0.614 |
| DBTSPL-MQ | 0.633 | 0.641 |

Fold 1 picks **DBTSPL-Q** (fold 2 full 0.648); on fold 1: 0.658 vs Selected 0.643 (+0.016).
Per task on fold 1 (pick minus Selected): antonym +0.21, synonym +0.03, singular-plural -0.00, present-past +0.14, country-capital +0.01, country-currency +0.01, landmark-country +0.02, park-country -0.02, product-company +0.02, next_item -0.03, prev_item -0.09, english-french +0.03, english-german -0.12, english-spanish +0.00
Fold 2 picks **DBTSPL-Q** (fold 1 full 0.658); on fold 2: 0.648 vs Selected 0.633 (+0.016).
Per task on fold 2 (pick minus Selected): antonym +0.25, synonym -0.00, singular-plural +0.06, present-past +0.26, country-capital +0.02, country-currency -0.01, landmark-country -0.02, park-country +0.01, product-company +0.00, next_item +0.00, prev_item +0.08, english-french -0.24, english-german -0.05, english-spanish -0.15

### shared

| combination | fold 1 | fold 2 |
|---|---|---|
| DBTM (Selected) | 0.617 | 0.627 |
| DBTM-O | 0.595 | 0.578 |
| DBTM-Q | 0.640 | 0.633 |

Fold 1 picks **DBTM-Q** (fold 2 full 0.633); on fold 1: 0.640 vs Selected 0.617 (+0.023).
Per task on fold 1 (pick minus Selected): antonym +0.17, synonym +0.25, singular-plural +0.09, present-past +0.09, country-capital -0.00, country-currency +0.08, landmark-country -0.02, park-country -0.02, product-company +0.01, next_item -0.03, prev_item +0.03, english-french -0.23, english-german -0.14, english-spanish +0.06
Fold 2 picks **DBTM-Q** (fold 1 full 0.640); on fold 2: 0.633 vs Selected 0.627 (+0.006).
Per task on fold 2 (pick minus Selected): antonym +0.15, synonym +0.22, singular-plural +0.07, present-past +0.05, country-capital -0.01, country-currency -0.06, landmark-country -0.00, park-country -0.02, product-company +0.00, next_item +0.03, prev_item +0.03, english-french -0.19, english-german -0.17, english-spanish -0.01

