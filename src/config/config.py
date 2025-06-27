import re
from pathlib import Path

import tiktoken
import torch

###############################################################################
TASK = "loan-adjudication"
TORCH_SEED = 217
MODEL_PROVIDER = "custom"  # "custom" or "huggingface"
###############################################################################

this_filepath = Path(__file__).resolve().parent
DATA_DIR = Path(this_filepath, "../../data/")
RESULTS_DIR_PATH = Path(this_filepath, "../../results")
CHECKPOINTS_DIR_PATH = Path(this_filepath, "../../checkpoints/gpt2")
FINETUNED_MODELS_DIR = Path(this_filepath, f"../../finetuned_models")
PRETRAINED_DIR = Path(this_filepath, f"../../pretrained_models")
LOSS_PLOT_FIG_FPATH = Path(RESULTS_DIR_PATH, "loss_plot.pdf")


###############################################################################
### DATA Preparation Configs ############
RULES_FPATH = Path(DATA_DIR, "fine_tune_llm_credit_rules.json")
DATASET_FPATH = Path(DATA_DIR, "loan_instruction_data_balanced.json")
INFERENCE_OUTPUT_DATA_FPATH = Path(
    DATA_DIR, "loan_instruction_data_inference_outputs.json"
)
NUM_SYNTHETIC_DATA = 1000
SYNTHETIC_DATA_PROPORTIONS = {"APPROVED": 0.34, "REJECTED": 0.33, "FLAG_REVIEW": 0.33}
TRAIN_VAL_TEST_SPLIT_RATIO = (0.7, 0.1, 0.2)


###############################################################################
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("----" * 30)
print(f"DEVICE = {DEVICE}")
print("----" * 30)

###############################################################################
###############################################################################
#### MODEL CONFIG #########################

BASE_MODEL_NAME = "gpt2"  # "google/gemma-2b-it"  # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # "google/gemma-2b-it"  # "meta-llama/Meta-Llama-3-8B"
CHOSEN_MODEL = "gpt2-medium (355M)"
PRETRAINED_MODELS_DIR = Path(PRETRAINED_DIR, f"{BASE_MODEL_NAME}")
FINETUNED_MODEL_FPATH = Path(
    FINETUNED_MODELS_DIR,
    f"{re.sub(r'[ ()]', '', CHOSEN_MODEL) }-sft-{TASK}.pth",
)

ALLOWED_GPT_SIZES = ("124M", "355M", "774M", "1558M")
BASE_GPT_WEIGHTS_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models"
BACKUP_GPT_WEIGHTS_URL = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
PRETRAINED_FILES_TO_DOWNLOAD = [
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
]

MODEL_EOT_TOKEN = "<|endoftext|>"
tokenizer = tiktoken.get_encoding(BASE_MODEL_NAME)
MODEL_EOT_ID = tokenizer.encode(
    text=MODEL_EOT_TOKEN, allowed_special={MODEL_EOT_TOKEN}
)[0]

MODEL_IGNORE_ID = -100  # this is also the PyTorch default
ALLOWED_MAX_LEN = 1024
MAX_NEW_GENERATED_TOKENS = 128
NUM_WORKERS = 0
BATCH_SIZE = 4  # depending on your GPU/CPU hardware resources

BASE_MODEL_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True,  # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_MODEL_CONFIG.update(model_configs[CHOSEN_MODEL])
MODEL_SIZE = CHOSEN_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

print("----" * 30)
print(f"BASE_MODEL_CONFIG = {BASE_MODEL_CONFIG}")
print("----" * 30)

###############################################################################
#### FINETUNING HYPERPARAMS #########################
RUN_INFERENCE_ON_TEST_DATA_AFTER_FINETUNING = True
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.1
LEARNING_RATE = 5e-5
MODE = "finetune"  # "inference"
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_DELTA = 0.0
###############################################################################
##### Sample inference ##############################
task_instruction = (
    f"Given the following applicant profile, decide whether to approve the following loan application and return your answer and reasons in the following format:"
    f"\n\nLABEL: One of the following (APPROVE or REJECT or FLAG_REVIEW)\nREASONS: A short explanation referring to the applicant's attributes.\n\n"
)
applicant_info = "### Input: \nThe applicant is a 24-year-old retired with a credit score of 691 and an annual income of $85731. Their debt-to-income ratio is 28.74%. They have been employed for 33 months. They are a US Citizen. Bankruptcy filed recently: Yes. Bank account verified: No. They have requested a loan of $48846."
user_input = f"{task_instruction}{applicant_info}\n\n### Response: "

###############################################################################
