# E9: E8's DB set on the complex tasks

Accuracy = mean over mean wordings (task: 7 rule wordings; control: control task's orig mean), at alpha 1 and 2. References: E2's J-head and Logit-head rule sets on the same items, alpha 1 and 2.

## check 1 (complex)

Baselines: Noun Class Agreement 0.00 (n=100), Lexical Category Inference 0.00 (n=100), Set Game 0.58 (n=100), Operator Function 0.00 (n=100), Tapatan 0.00 (n=100)

| ('task', '') | ('heads', '') | ('control', 1.0) | ('control', 2.0) | ('task', 1.0) | ('task', 2.0) |
|---|---|---|---|---|---|
| lexical_category_inference/easy | DB | 0.32 | 0.41 | 0.03 | 0.13 |
| lexical_category_inference/easy | jhead | 0.00 | 0.00 | 0.00 | 0.00 |
| lexical_category_inference/easy | logithead | 0.00 | 0.00 | 0.00 | 0.00 |
| noun_class_agreement/easy | DB | 0.02 | 0.00 | 0.06 | 0.21 |
| noun_class_agreement/easy | jhead | 0.00 | 0.00 | 0.00 | 0.00 |
| noun_class_agreement/easy | logithead | 0.00 | 0.00 | 0.00 | 0.00 |
| operator_func/easy | DB | 0.00 | 0.00 | 0.00 | 0.00 |
| operator_func/easy | jhead | 0.00 | 0.00 | 0.00 | 0.00 |
| operator_func/easy | logithead | 0.00 | 0.00 | 0.00 | 0.00 |
| set_game/easy | DB | 0.62 | 0.39 | 0.59 | 0.57 |
| set_game/easy | jhead | 0.63 | 0.58 | 0.60 | 0.58 |
| set_game/easy | logithead | 0.55 | 0.61 | 0.60 | 0.59 |
| tapatan/easy | DB | 0.01 | 0.01 | 0.01 | 0.01 |
| tapatan/easy | jhead | 0.00 | 0.00 | 0.00 | 0.00 |
| tapatan/easy | logithead | 0.00 | 0.00 | 0.00 | 0.01 |

## check 2 (complex-wrong)

Baselines: Noun Class Agreement 0.00 (n=100), Lexical Category Inference 0.00 (n=100), Set Game 0.00 (n=100), Tapatan 0.00 (n=100)

| ('task', '') | ('heads', '') | ('control', 1.0) | ('control', 2.0) | ('task', 1.0) | ('task', 2.0) |
|---|---|---|---|---|---|
| lexical_category_inference/easy | DB | 0.28 | 0.38 | 0.33 | 0.41 |
| lexical_category_inference/easy | jhead | 0.12 | 0.27 | 0.12 | 0.27 |
| lexical_category_inference/easy | logithead | 0.04 | 0.21 | 0.09 | 0.28 |
| noun_class_agreement/easy | DB | 0.35 | 0.58 | 0.36 | 0.58 |
| noun_class_agreement/easy | jhead | 0.09 | 0.16 | 0.03 | 0.05 |
| noun_class_agreement/easy | logithead | 0.08 | 0.13 | 0.04 | 0.05 |
| set_game/easy | DB | 0.20 | 0.24 | 0.21 | 0.21 |
| set_game/easy | jhead | 0.18 | 0.23 | 0.21 | 0.22 |
| set_game/easy | logithead | 0.24 | 0.23 | 0.22 | 0.22 |
| tapatan/easy | DB | 0.17 | 0.31 | 0.17 | 0.19 |
| tapatan/easy | jhead | 0.16 | 0.15 | 0.15 | 0.16 |
| tapatan/easy | logithead | 0.13 | 0.17 | 0.13 | 0.13 |

## Answer checks

| family | task | heads | source | alpha | n | answered | no_answer | accuracy | top_answer | top_share | one_label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| complex | lexical_category_inference/easy | DB | control | 1.00 | 100 | 0.51 | 0.49 | 0.32 | Yes | 0.37 | 0.27 |
| complex | lexical_category_inference/easy | DB | control | 2.00 | 100 | 1.00 | 0.00 | 0.41 | Yes | 1.00 | 0.59 |
| complex | lexical_category_inference/easy | DB | task | 1.00 | 700 | 0.05 | 0.95 | 0.03 | No | 0.04 | 0.03 |
| complex | lexical_category_inference/easy | DB | task | 2.00 | 700 | 0.28 | 0.72 | 0.13 | Yes | 0.28 | 0.17 |
| complex | noun_class_agreement/easy | DB | control | 1.00 | 100 | 0.05 | 0.95 | 0.02 | Yes | 0.05 | 0.03 |
| complex | noun_class_agreement/easy | DB | control | 2.00 | 100 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | noun_class_agreement/easy | DB | task | 1.00 | 700 | 0.13 | 0.87 | 0.06 | Yes | 0.13 | 0.06 |
| complex | noun_class_agreement/easy | DB | task | 2.00 | 700 | 0.45 | 0.55 | 0.21 | Yes | 0.45 | 0.24 |
| complex | operator_func/easy | DB | control | 1.00 | 100 | 0.55 | 0.45 | 0.00 |  |  |  |
| complex | operator_func/easy | DB | control | 2.00 | 100 | 0.45 | 0.55 | 0.00 |  |  |  |
| complex | operator_func/easy | DB | task | 1.00 | 700 | 0.78 | 0.22 | 0.00 |  |  |  |
| complex | operator_func/easy | DB | task | 2.00 | 700 | 0.50 | 0.50 | 0.00 |  |  |  |
| complex | set_game/easy | DB | control | 1.00 | 100 | 0.78 | 0.22 | 0.62 |  |  |  |
| complex | set_game/easy | DB | control | 2.00 | 100 | 0.60 | 0.40 | 0.39 |  |  |  |
| complex | set_game/easy | DB | task | 1.00 | 700 | 0.77 | 0.23 | 0.59 |  |  |  |
| complex | set_game/easy | DB | task | 2.00 | 700 | 0.75 | 0.25 | 0.57 |  |  |  |
| complex | tapatan/easy | DB | control | 1.00 | 100 | 0.01 | 0.99 | 0.01 | continue | 0.01 | 0.01 |
| complex | tapatan/easy | DB | control | 2.00 | 100 | 0.02 | 0.98 | 0.01 | continue | 0.02 | 0.01 |
| complex | tapatan/easy | DB | task | 1.00 | 700 | 0.01 | 0.99 | 0.01 | continue | 0.01 | 0.01 |
| complex | tapatan/easy | DB | task | 2.00 | 700 | 0.01 | 0.99 | 0.01 | continue | 0.01 | 0.01 |
| complex-wrong | lexical_category_inference/easy | DB | control | 1.00 | 100 | 1.00 | 0.00 | 0.28 | No | 1.00 | 0.72 |
| complex-wrong | lexical_category_inference/easy | DB | control | 2.00 | 100 | 1.00 | 0.00 | 0.38 | No | 0.84 | 0.72 |
| complex-wrong | lexical_category_inference/easy | DB | task | 1.00 | 700 | 1.00 | 0.00 | 0.33 | No | 0.81 | 0.72 |
| complex-wrong | lexical_category_inference/easy | DB | task | 2.00 | 700 | 1.00 | 0.00 | 0.41 | No | 0.71 | 0.72 |
| complex-wrong | noun_class_agreement/easy | DB | control | 1.00 | 100 | 1.00 | 0.00 | 0.35 | No | 0.93 | 0.58 |
| complex-wrong | noun_class_agreement/easy | DB | control | 2.00 | 100 | 1.00 | 0.00 | 0.58 | Yes | 1.00 | 0.58 |
| complex-wrong | noun_class_agreement/easy | DB | task | 1.00 | 700 | 1.00 | 0.00 | 0.36 | No | 0.94 | 0.58 |
| complex-wrong | noun_class_agreement/easy | DB | task | 2.00 | 700 | 1.00 | 0.00 | 0.58 | Yes | 1.00 | 0.58 |
| complex-wrong | set_game/easy | DB | control | 1.00 | 100 | 0.73 | 0.27 | 0.20 |  |  |  |
| complex-wrong | set_game/easy | DB | control | 2.00 | 100 | 0.70 | 0.30 | 0.24 |  |  |  |
| complex-wrong | set_game/easy | DB | task | 1.00 | 700 | 0.76 | 0.24 | 0.21 |  |  |  |
| complex-wrong | set_game/easy | DB | task | 2.00 | 700 | 0.78 | 0.22 | 0.21 |  |  |  |
| complex-wrong | tapatan/easy | DB | control | 1.00 | 100 | 0.85 | 0.15 | 0.17 | A win | 0.53 | 0.35 |
| complex-wrong | tapatan/easy | DB | control | 2.00 | 100 | 0.88 | 0.12 | 0.31 | A win | 0.46 | 0.39 |
| complex-wrong | tapatan/easy | DB | task | 1.00 | 700 | 0.84 | 0.16 | 0.17 | A win | 0.45 | 0.36 |
| complex-wrong | tapatan/easy | DB | task | 2.00 | 700 | 0.88 | 0.12 | 0.19 | A win | 0.56 | 0.38 |
| complex | lexical_category_inference/easy | jhead | control | 1.00 | 100 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | lexical_category_inference/easy | jhead | control | 2.00 | 100 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | lexical_category_inference/easy | jhead | task | 1.00 | 700 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | lexical_category_inference/easy | jhead | task | 2.00 | 700 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | lexical_category_inference/easy | logithead | control | 1.00 | 100 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | lexical_category_inference/easy | logithead | control | 2.00 | 100 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | lexical_category_inference/easy | logithead | task | 1.00 | 700 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | lexical_category_inference/easy | logithead | task | 2.00 | 700 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | noun_class_agreement/easy | jhead | control | 1.00 | 100 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | noun_class_agreement/easy | jhead | control | 2.00 | 100 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | noun_class_agreement/easy | jhead | task | 1.00 | 700 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | noun_class_agreement/easy | jhead | task | 2.00 | 700 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | noun_class_agreement/easy | logithead | control | 1.00 | 100 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | noun_class_agreement/easy | logithead | control | 2.00 | 100 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | noun_class_agreement/easy | logithead | task | 1.00 | 700 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | noun_class_agreement/easy | logithead | task | 2.00 | 700 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | operator_func/easy | jhead | control | 1.00 | 100 | 0.42 | 0.58 | 0.00 |  |  |  |
| complex | operator_func/easy | jhead | control | 2.00 | 100 | 0.44 | 0.56 | 0.00 |  |  |  |
| complex | operator_func/easy | jhead | task | 1.00 | 700 | 0.54 | 0.46 | 0.00 |  |  |  |
| complex | operator_func/easy | jhead | task | 2.00 | 700 | 0.59 | 0.41 | 0.00 |  |  |  |
| complex | operator_func/easy | logithead | control | 1.00 | 100 | 0.48 | 0.52 | 0.00 |  |  |  |
| complex | operator_func/easy | logithead | control | 2.00 | 100 | 0.58 | 0.42 | 0.00 |  |  |  |
| complex | operator_func/easy | logithead | task | 1.00 | 700 | 0.53 | 0.47 | 0.00 |  |  |  |
| complex | operator_func/easy | logithead | task | 2.00 | 700 | 0.52 | 0.48 | 0.00 |  |  |  |
| complex | set_game/easy | jhead | control | 1.00 | 100 | 0.77 | 0.23 | 0.63 |  |  |  |
| complex | set_game/easy | jhead | control | 2.00 | 100 | 0.78 | 0.22 | 0.58 |  |  |  |
| complex | set_game/easy | jhead | task | 1.00 | 700 | 0.78 | 0.22 | 0.60 |  |  |  |
| complex | set_game/easy | jhead | task | 2.00 | 700 | 0.75 | 0.25 | 0.58 |  |  |  |
| complex | set_game/easy | logithead | control | 1.00 | 100 | 0.75 | 0.25 | 0.55 |  |  |  |
| complex | set_game/easy | logithead | control | 2.00 | 100 | 0.76 | 0.24 | 0.61 |  |  |  |
| complex | set_game/easy | logithead | task | 1.00 | 700 | 0.80 | 0.20 | 0.60 |  |  |  |
| complex | set_game/easy | logithead | task | 2.00 | 700 | 0.80 | 0.20 | 0.59 |  |  |  |
| complex | tapatan/easy | jhead | control | 1.00 | 100 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | tapatan/easy | jhead | control | 2.00 | 100 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | tapatan/easy | jhead | task | 1.00 | 700 | 0.01 | 0.99 | 0.00 | continue | 0.01 | 0.00 |
| complex | tapatan/easy | jhead | task | 2.00 | 700 | 0.00 | 1.00 | 0.00 | continue | 0.00 | 0.00 |
| complex | tapatan/easy | logithead | control | 1.00 | 100 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | tapatan/easy | logithead | control | 2.00 | 100 | 0.00 | 1.00 | 0.00 |  | 0.00 | 0.00 |
| complex | tapatan/easy | logithead | task | 1.00 | 700 | 0.00 | 1.00 | 0.00 | continue | 0.00 | 0.00 |
| complex | tapatan/easy | logithead | task | 2.00 | 700 | 0.01 | 0.99 | 0.01 | continue | 0.01 | 0.01 |
| complex-wrong | lexical_category_inference/easy | jhead | control | 1.00 | 100 | 1.00 | 0.00 | 0.12 | No | 0.84 | 0.72 |
| complex-wrong | lexical_category_inference/easy | jhead | control | 2.00 | 100 | 1.00 | 0.00 | 0.27 | No | 0.99 | 0.72 |
| complex-wrong | lexical_category_inference/easy | jhead | task | 1.00 | 700 | 1.00 | 0.00 | 0.12 | No | 0.77 | 0.72 |
| complex-wrong | lexical_category_inference/easy | jhead | task | 2.00 | 700 | 1.00 | 0.00 | 0.27 | No | 0.99 | 0.72 |
| complex-wrong | lexical_category_inference/easy | logithead | control | 1.00 | 100 | 1.00 | 0.00 | 0.04 | No | 0.68 | 0.72 |
| complex-wrong | lexical_category_inference/easy | logithead | control | 2.00 | 100 | 1.00 | 0.00 | 0.21 | No | 0.93 | 0.72 |
| complex-wrong | lexical_category_inference/easy | logithead | task | 1.00 | 700 | 1.00 | 0.00 | 0.09 | No | 0.78 | 0.72 |
| complex-wrong | lexical_category_inference/easy | logithead | task | 2.00 | 700 | 1.00 | 0.00 | 0.28 | No | 1.00 | 0.72 |
| complex-wrong | noun_class_agreement/easy | jhead | control | 1.00 | 100 | 1.00 | 0.00 | 0.09 | No | 0.67 | 0.58 |
| complex-wrong | noun_class_agreement/easy | jhead | control | 2.00 | 100 | 1.00 | 0.00 | 0.16 | No | 0.74 | 0.58 |
| complex-wrong | noun_class_agreement/easy | jhead | task | 1.00 | 700 | 1.00 | 0.00 | 0.03 | No | 0.59 | 0.58 |
| complex-wrong | noun_class_agreement/easy | jhead | task | 2.00 | 700 | 1.00 | 0.00 | 0.05 | No | 0.63 | 0.58 |
| complex-wrong | noun_class_agreement/easy | logithead | control | 1.00 | 100 | 1.00 | 0.00 | 0.08 | No | 0.66 | 0.58 |
| complex-wrong | noun_class_agreement/easy | logithead | control | 2.00 | 100 | 1.00 | 0.00 | 0.13 | No | 0.71 | 0.58 |
| complex-wrong | noun_class_agreement/easy | logithead | task | 1.00 | 700 | 1.00 | 0.00 | 0.04 | No | 0.55 | 0.58 |
| complex-wrong | noun_class_agreement/easy | logithead | task | 2.00 | 700 | 1.00 | 0.00 | 0.05 | No | 0.53 | 0.58 |
| complex-wrong | set_game/easy | jhead | control | 1.00 | 100 | 0.74 | 0.26 | 0.18 |  |  |  |
| complex-wrong | set_game/easy | jhead | control | 2.00 | 100 | 0.76 | 0.24 | 0.23 |  |  |  |
| complex-wrong | set_game/easy | jhead | task | 1.00 | 700 | 0.74 | 0.26 | 0.21 |  |  |  |
| complex-wrong | set_game/easy | jhead | task | 2.00 | 700 | 0.78 | 0.22 | 0.22 |  |  |  |
| complex-wrong | set_game/easy | logithead | control | 1.00 | 100 | 0.73 | 0.27 | 0.24 |  |  |  |
| complex-wrong | set_game/easy | logithead | control | 2.00 | 100 | 0.70 | 0.30 | 0.23 |  |  |  |
| complex-wrong | set_game/easy | logithead | task | 1.00 | 700 | 0.77 | 0.23 | 0.22 |  |  |  |
| complex-wrong | set_game/easy | logithead | task | 2.00 | 700 | 0.74 | 0.26 | 0.22 |  |  |  |
| complex-wrong | tapatan/easy | jhead | control | 1.00 | 100 | 0.86 | 0.14 | 0.16 | A win | 0.55 | 0.38 |
| complex-wrong | tapatan/easy | jhead | control | 2.00 | 100 | 0.90 | 0.10 | 0.15 | A win | 0.54 | 0.39 |
| complex-wrong | tapatan/easy | jhead | task | 1.00 | 700 | 0.86 | 0.14 | 0.15 | A win | 0.52 | 0.37 |
| complex-wrong | tapatan/easy | jhead | task | 2.00 | 700 | 0.85 | 0.15 | 0.16 | A win | 0.53 | 0.35 |
| complex-wrong | tapatan/easy | logithead | control | 1.00 | 100 | 0.84 | 0.16 | 0.13 | A win | 0.46 | 0.36 |
| complex-wrong | tapatan/easy | logithead | control | 2.00 | 100 | 0.89 | 0.11 | 0.17 | A win | 0.60 | 0.39 |
| complex-wrong | tapatan/easy | logithead | task | 1.00 | 700 | 0.86 | 0.14 | 0.13 | A win | 0.53 | 0.37 |
| complex-wrong | tapatan/easy | logithead | task | 2.00 | 700 | 0.85 | 0.15 | 0.13 | A win | 0.52 | 0.34 |
