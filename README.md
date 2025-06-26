# `Loan Adjudicator`: Fine-Tuning a Language Model for Loan Adjudication

```text
Author: Vinay Kumar
Year: 2025
```

## Objective

The goal of this project is to simulate a dataset of personal loan applications based on a well-defined credit ruleset, and fine-tune a large language model to:
- Predict the loan decision (`APPROVED / REJECTED / FLAG_REVIEW`)
- Provide human-readable reasons for the decision

---

## Synthetic Dataset Generation

### Ruleset Source

The dataset is derived from a JSON ruleset file containing credit evaluation rules such as:
- Minimum age (`>= 18`)
- Minimum credit score (`>= 670`)
- Minimum annual income (`>= $30,000`)
- Maximum debt-to-income ratio (`<= 40%`)
- Employment status, employment duration
- Residency status
- Bankruptcy history
- Loan amount proportional to income
- Bank account verification

Each rule includes:
- Rule ID and description
- Applicant field to evaluate
- Operator and threshold value
- Severity and action on fail (`REJECT, FLAG_REVIEW`)

### Applicant Simulation

Synthetic applicant profiles are generated randomly with the following attributes:
- Age, credit score, annual income
- Debt-to-income ratio
- Employment status & duration
- Residency status
- Bankruptcy and bank account presence
- Loan amount requested

### Rule Evaluation Logic

Each applicant profile is checked against all rules using:
- Direct comparison (`>=`, `<=`, `in`, `is`)
- Special evaluation for dynamic fields (e.g., loan amount ≤ 0.5 × income)

The final label is chosen based on the highest-severity failed rule:
- If any rule fails with `REJECT`, label is `REJECTED`
- Else if any rule fails with `FLAG_REVIEW`, label is `FLAG_REVIEW`
- Otherwise, label is `APPROVED`

### Label Balancing

To ensure a balanced dataset for training:
- Target label distribution: 
  - APPROVED: `34%`
  - REJECTED: `33%`
  - FLAG_REVIEW: `33%`
- Samples are selectively added until each label count reaches its target

---

## Model Fine-Tuning

### Base Model

`gpt2` was used as the base model due to:
- Lightweight architecture suitable for CPU and CUDA backends
- The goal of this project was also to implement the architecture for in-depth understanding and demonstration, hence I chose GPT2 which is both capable, provides multiple versions of the models and is suitable for demostration.
- Open license and community support from OpenAI

### Training Setup

- Framework: `PyTorch` 
- Tokenizer: `tiktoken`
- Input format: Instruction-tuned prompt structure:
  - `instruction`: Describes the decision task
  - `input`: Natural language summary of the applicant
  - `output`: Expected response in structured format
- Checkpointing enabled for resume/recovery after every epoch.
- Dataset split: `train`, `validation`, and `test`
- `src/config/config.py` holds all settings for synthetic dataset generation and model hyperparameters.

---

## Evaluation & Results

- Final train loss: `...`
- Final validation loss: `...`
- Final test loss: `...`
- The plot for finetuning routine can be found in `./results` folder
- Gradients and learning rate stabilized as training progressed
- The model learned to generate interpretable, rule-based decisions effectively. The generated output for `test` daatset can be found in `./data/loan_instruction_data_inference_outputs.json`

---

## Inference
The inference stage can be activated by setting `MODE = "inference"` inside `src/config/config.py`and then running `bash ./tools/run_main_routine.sh`