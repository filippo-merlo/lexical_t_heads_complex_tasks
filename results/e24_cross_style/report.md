# E24: rule-built sets with example means, and sets built on example prompts

Check 1, held-out items, cross-fitted alpha in {1, 2}, mean of two folds and 14 tasks.

## Step A: best rule-built sets

| set | scope | rule_task | rule_control | example_task | example_control |
|---|---|---|---|---|---|
| DBTMS100 | task | 0.683 | 0.022 | 0.643 | 0.027 |
| DBMN60 | task | 0.664 | 0.024 | 0.486 | 0.026 |
| DBT40 | task | 0.553 | 0.017 | 0.449 | 0.017 |
| causal120 | task | 0.614 | 0.035 | 0.453 | 0.043 |
| random_dbt100 | task | 0.006 | -- | 0.328 | -- |
| random_ex_db100 | task | 0.023 | -- | 0.032 | -- |
| shared_causal100_nogen | shared | 0.711 | 0.013 | 0.498 | 0.013 |
| shared_causal80_nogen | shared | 0.691 | 0.02 | 0.515 | 0.019 |
| shared_DBTMS120 | shared | 0.711 | 0.024 | 0.582 | 0.031 |
| shared_DBMN80 | shared | 0.698 | 0.017 | 0.544 | 0.019 |
| random_shared_dbt100 | shared | 0.008 | -- | 0.34 | -- |
| random_ex_shared_db100 | shared | 0.008 | -- | 0.005 | -- |

## Steps B and C: built on rule or example prompts x rule or example means

| set | scope | rule_built_rule | rule_built_example | example_built_rule | example_built_example | overlap |
|---|---|---|---|---|---|---|
| D-40 | task | 0.369 | 0.309 | 0.307 | 0.509 | 0.38 (0.07) |
| B-40 | task | 0.224 | 0.097 | 0.088 | 0.577 | 0.43 (0.11) |
| DB-40 | task | 0.453 | 0.276 | 0.401 | 0.69 | 0.40 (0.13) |
| DBMS-40 | task | 0.508 | 0.283 | 0.389 | 0.635 | 0.25 (0.11) |
| Random-40 | task | -- | -- | 0.016 | 0.014 | -- |
| D-100 | task | 0.4 | 0.593 | 0.483 | 0.731 | 0.43 (0.14) |
| B-100 | task | 0.225 | 0.038 | 0.123 | 0.585 | 0.49 (0.22) |
| DB-100 | task | 0.269 | 0.097 | 0.332 | 0.539 | 0.48 (0.22) |
| DBMS-100 | task | 0.664 | 0.599 | 0.539 | 0.774 | 0.40 (0.23) |
| Random-100 | task | -- | -- | 0.023 | 0.032 | -- |
| D-40 | shared | 0.291 | 0.264 | 0.251 | 0.563 | 0.34 (0.08) |
| B-40 | shared | 0.251 | 0.088 | 0.011 | 0.397 | 0.45 (0.15) |
| DB-40 | shared | 0.472 | 0.285 | 0.468 | 0.78 | 0.55 (0.15) |
| DBMS-40 | shared | 0.577 | 0.313 | 0.243 | 0.347 | 0.35 (0.13) |
| Random-40 | shared | -- | -- | 0.013 | 0.053 | -- |
| D-100 | shared | 0.219 | 0.556 | 0.395 | 0.579 | 0.47 (0.18) |
| B-100 | shared | 0.443 | 0.147 | 0.225 | 0.738 | 0.51 (0.27) |
| DB-100 | shared | 0.454 | 0.315 | 0.518 | 0.716 | 0.57 (0.25) |
| DBMS-100 | shared | 0.67 | 0.467 | 0.524 | 0.756 | 0.40 (0.27) |
| Random-100 | shared | -- | -- | 0.008 | 0.005 | -- |

## Overlap of rule- and example-built sets (share of N; chance: layer-matched)

| scope | combo | n | share | chance |
|---|---|---|---|---|
| shared | B | 20 | 0.45 | 0.095 |
| shared | B | 40 | 0.45 | 0.145 |
| shared | B | 60 | 0.442 | 0.182 |
| shared | B | 80 | 0.475 | 0.244 |
| shared | B | 100 | 0.51 | 0.268 |
| shared | B | 120 | 0.538 | 0.296 |
| shared | D | 20 | 0.225 | 0.057 |
| shared | D | 40 | 0.337 | 0.084 |
| shared | D | 60 | 0.392 | 0.118 |
| shared | D | 80 | 0.481 | 0.15 |
| shared | D | 100 | 0.47 | 0.175 |
| shared | D | 120 | 0.533 | 0.205 |
| shared | DB | 20 | 0.35 | 0.101 |
| shared | DB | 40 | 0.55 | 0.148 |
| shared | DB | 60 | 0.583 | 0.202 |
| shared | DB | 80 | 0.544 | 0.226 |
| shared | DB | 100 | 0.57 | 0.245 |
| shared | DB | 120 | 0.612 | 0.291 |
| shared | DBMS | 20 | 0.45 | 0.097 |
| shared | DBMS | 40 | 0.35 | 0.129 |
| shared | DBMS | 60 | 0.317 | 0.171 |
| shared | DBMS | 80 | 0.381 | 0.216 |
| shared | DBMS | 100 | 0.405 | 0.266 |
| shared | DBMS | 120 | 0.471 | 0.309 |
| task | B | 20 | 0.464 | 0.074 |
| task | B | 40 | 0.43 | 0.112 |
| task | B | 60 | 0.451 | 0.15 |
| task | B | 80 | 0.464 | 0.19 |
| task | B | 100 | 0.49 | 0.225 |
| task | B | 120 | 0.499 | 0.261 |
| task | D | 20 | 0.3 | 0.043 |
| task | D | 40 | 0.383 | 0.073 |
| task | D | 60 | 0.389 | 0.098 |
| task | D | 80 | 0.413 | 0.122 |
| task | D | 100 | 0.427 | 0.144 |
| task | D | 120 | 0.454 | 0.169 |
| task | DB | 20 | 0.332 | 0.085 |
| task | DB | 40 | 0.396 | 0.129 |
| task | DB | 60 | 0.432 | 0.162 |
| task | DB | 80 | 0.458 | 0.193 |
| task | DB | 100 | 0.484 | 0.222 |
| task | DB | 120 | 0.517 | 0.247 |
| task | DBMS | 20 | 0.223 | 0.065 |
| task | DBMS | 40 | 0.246 | 0.11 |
| task | DBMS | 60 | 0.305 | 0.155 |
| task | DBMS | 80 | 0.357 | 0.192 |
| task | DBMS | 100 | 0.402 | 0.23 |
| task | DBMS | 120 | 0.432 | 0.267 |

## Task-word lists: words shared by the 5-shot and the rule list (of 30)

| task | shared_words |
|---|---|
| antonym | 7.0 |
| country-capital | 15.5 |
| country-currency | 15.0 |
| english-french | 12.5 |
| english-german | 16.0 |
| english-spanish | 15.0 |
| landmark-country | 16.5 |
| next_item | 11.0 |
| park-country | 16.0 |
| present-past | 9.0 |
| prev_item | 9.0 |
| product-company | 13.0 |
| singular-plural | 8.0 |
| synonym | 9.5 |
