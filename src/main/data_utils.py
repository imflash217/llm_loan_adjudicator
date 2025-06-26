import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

# from datasets import Dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import config as cfg


def download_data_as_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def get_train_valid_test_split(data, train_pct, val_pct, test_pct):
    assert train_pct + val_pct + test_pct == 1.0

    train_portion = int(len(data) * train_pct)
    test_portion = int(len(data) * test_pct)
    val_portion = int(len(data) * val_pct)

    train_data = data[:train_portion]
    val_data = data[train_portion : train_portion + val_portion]
    test_data = data[train_portion + val_portion :]

    assert test_portion == len(test_data)
    return train_data, val_data, test_data


def format_input(item):
    instruction_text = f"### Instruction:\n{item['instruction']}"

    input_text = f"\n\n### Applicant Profile:\n{item['input']}" if item["input"] else ""
    return instruction_text + input_text


class LoanDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for item in data:
            instruction_plus_input = format_input(item)
            response_text = f"\n\n### Response:\n{item['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=cfg.MODEL_EOT_ID,
    ignore_index=cfg.MODEL_IGNORE_ID,
    allowed_max_length=cfg.ALLOWED_MAX_LEN,
    device=cfg.DEVICE,
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None and allowed_max_length > 0:
            allowed_max_length = min(batch_max_length, allowed_max_length)
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def get_dataloaders(data, collate_fn, split=(0.7, 0.1, 0.2)):
    train_data, val_data, test_data = get_train_valid_test_split(data, *split)
    train_dataset = LoanDataset(train_data, cfg.tokenizer)
    val_dataset = LoanDataset(val_data, cfg.tokenizer)
    test_dataset = LoanDataset(test_data, cfg.tokenizer)

    train_cfg = {
        "batch_size": cfg.BATCH_SIZE,
        "collate_fn": collate_fn,
        "shuffle": True,
        "drop_last": True,
        "num_workers": cfg.NUM_WORKERS,
    }
    test_cfg = {
        "batch_size": cfg.BATCH_SIZE,
        "collate_fn": collate_fn,
        "shuffle": False,
        "drop_last": False,
        "num_workers": cfg.NUM_WORKERS,
    }

    train_loader = DataLoader(train_dataset, **train_cfg)
    val_loader = DataLoader(val_dataset, **test_cfg)
    test_loader = DataLoader(test_dataset, **test_cfg)

    return train_loader, val_loader, test_loader


##############################################################################
######## synethetic data generation ##########################################


# Load rules from the provided JSON file
def load_rules(json_path: str | Path) -> List[Dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["personal_loan_credit_rules"]["rules"]


# Generate a realistic synthetic applicant
def generate_applicant() -> Dict:
    return {
        "applicant": {
            "age": random.randint(16, 100),
            "credit_score": random.randint(600, 800),
            "annual_income_usd": random.randint(10000, 300000),
            "debt_to_income_ratio_percent": round(random.uniform(10, 60), 2),
            "employment_status": random.choice(
                [
                    "employed_full_time",
                    "employed_part_time",
                    "self_employed",
                    "retired",
                    "unemployed",
                    "student",
                ]
            ),
            "current_employment_duration_months": random.randint(0, 36),
            "residency_status": random.choice(
                [
                    "US_Citizen",
                    "Permanent_Resident",
                    "Temporary_Resident",
                    "Non_Resident",
                    "Alien",
                ]
            ),
            "has_bankruptcy_recent": random.choice([True, False]),
            "has_verifiable_bank_account": random.choice([True, False]),
        },
        "loan_application": {"requested_amount_usd": random.randint(5000, 50000)},
    }


# Evaluate an applicant against the ruleset
def evaluate_applicant(profile: Dict, rules: List[Dict]) -> tuple[str, List[str]]:
    reasons = []
    decisions = []

    for rule in rules:
        field_parts = rule["field"].split(".")
        value = profile
        for part in field_parts:
            value = value.get(part, None)

        op = rule["operator"]
        expected = rule.get("value")

        if "value_field_multiplier" in rule:
            multiplier_field = rule["value_field_multiplier"].split(".")
            base = profile
            for part in multiplier_field:
                base = base.get(part, None)
            # print(base)
            expected = base * rule["multiplier_value"]

        passed = True
        if op == ">=" and not (value >= expected):
            passed = False
        elif op == "<=" and not (value <= expected):
            passed = False
        elif op == "is" and value is not expected:
            passed = False
        elif op == "in" and value not in expected:
            passed = False

        if not passed:
            decisions.append(rule["action_on_fail"])
            reasons.append(rule["description"])

    if "REJECT" in decisions:
        label = "REJECTED"
    elif "FLAG_REVIEW" in decisions:
        label = "FLAG_REVIEW"
    else:
        label = "APPROVED"

    return label, reasons


# Create a natural language profile summary
def format_profile(profile: Dict) -> str:
    p = profile["applicant"]
    l = profile["loan_application"]
    return (
        f"The applicant is a {p['age']}-year-old {p['employment_status'].replace('_', ' ')} with a credit score of {p['credit_score']} "
        f"and an annual income of ${p['annual_income_usd']}. Their debt-to-income ratio is {p['debt_to_income_ratio_percent']}%. "
        f"They have been employed for {p['current_employment_duration_months']} months. They are a {p['residency_status'].replace('_', ' ')}. "
        f"Bankruptcy filed recently: {'Yes' if p['has_bankruptcy_recent'] else 'No'}. Bank account verified: {'Yes' if p['has_verifiable_bank_account'] else 'No'}. "
        f"They have requested a loan of ${l['requested_amount_usd']}."
    )


# Generate a dataset with custom label proportions
def generate_dataset_proportional(
    rules_path: str | Path, n: int = 100, proportions: Dict[str, float] = None
) -> List[Dict]:
    if proportions is None:
        proportions = {"APPROVED": 0.5, "REJECTED": 0.25, "FLAG_REVIEW": 0.25}

    rules = load_rules(rules_path)
    dataset = []
    label_counts = {label: 0 for label in proportions.keys()}
    per_label_target = {label: int(n * frac) for label, frac in proportions.items()}

    while any(label_counts[label] < per_label_target[label] for label in proportions):
        profile = generate_applicant()
        label, reasons = evaluate_applicant(profile, rules)
        if label not in proportions or label_counts[label] >= per_label_target[label]:
            continue

        input_text = format_profile(profile)
        reasons = (
            " ".join(reasons)
            if reasons
            else "The applicant meets all required conditions."
        )

        dataset.append(
            {
                "instruction": (
                    "Given the following applicant profile, decide whether to approve the following loan application and return your answer and reasons in the following format:"
                    "\n\nLABEL: One of the following (APPROVE or REJECT or FLAG_REVIEW)"
                    "\nREASONS: A short explanation referring to the applicant's attributes"
                ),
                "input": input_text,
                "output": f"LABEL: {label}\n\nREASONS: {reasons}",
                "label": label,
                "reasons": reasons,
            }
        )

        label_counts[label] += 1

    print("Label distribution:", label_counts)
    # Shuffle dataset to ensure randomness
    random.shuffle(dataset)
    return dataset


# Save dataset to JSON file
def save_json(data: List[Dict], filename: str | Path):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


# def to_hf_dataset(data: List[Dict]) -> Dataset:
#     return Dataset.from_pandas(pd.DataFrame(data))


def plot_label_distribution(data: List[Dict]):
    df = pd.DataFrame(data)
    counts = df["label"].value_counts()
    counts.plot(kind="bar", title="Label Distribution", xlabel="Label", ylabel="Count")
    plt.show()
