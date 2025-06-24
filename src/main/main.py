import json
import os
import time
from functools import partial

import torch
from tqdm import tqdm

from src.config import config as cfg
from src.main import data_utils as ux
from src.main import models as mx
from src.main.model_utils import (
    generate,
    plot_losses,
    text_to_token_ids,
    token_ids_to_text,
    train_model,
)


def run_finetuning_model(train_dl, val_dl, test_data, val_data):
    model = mx.get_pretrained_model()
    model.to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    train_losses, val_losses, tokens_seen = train_model(
        model,
        train_dl,
        val_dl,
        optimizer,
        cfg.DEVICE,
        num_epochs=cfg.NUM_EPOCHS,
        eval_freq=5,
        eval_iter=5,
        start_context=ux.format_input(val_data[0]),
        tokenizer=cfg.tokenizer,
    )
    epochs_tensor = torch.linspace(0, cfg.NUM_EPOCHS, len(train_losses))

    plot_losses(
        epochs_seen=epochs_tensor,
        tokens_seen=tokens_seen,
        train_losses=train_losses,
        val_losses=val_losses,
    )

    if cfg.RUN_INFERENCE_ON_TEST_DATA_AFTER_FINETUNING:
        for i, item in tqdm(enumerate(test_data), total=len(test_data)):
            input_text = ux.format_input(item)
            token_ids = generate(
                model=model,
                idx=text_to_token_ids(input_text, cfg.tokenizer).to(cfg.DEVICE),
                max_new_tokens=cfg.MAX_NEW_GENERATED_TOKENS,
                context_size=cfg.BASE_MODEL_CONFIG["context_length"],
                eos_id=cfg.MODEL_EOT_ID,
            )
            generated_text = token_ids_to_text(token_ids, cfg.tokenizer)
            response_text = (
                generated_text[len(input_text) :].replace("### Response:", "").strip()
            )
            test_data[i]["model_response"] = response_text

        with open(cfg.INFERENCE_OUTPUT_DATA_FPATH, "w") as file:
            json.dump(test_data, file, indent=4)


if __name__ == "__main__":
    os.makedirs(cfg.PRETRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINTS_DIR_PATH, exist_ok=True)
    os.makedirs(cfg.FINETUNED_MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR_PATH, exist_ok=True)

    if cfg.MODE == "finetune":
        start_time = time.time()
        torch.manual_seed(123)

        data = ux.load_data(file_path=cfg.DATASET_FPATH)
        train_data, val_data, test_data = ux.get_train_valid_test_split(
            data, *cfg.TRAIN_VAL_TEST_SPLIT_RATIO
        )
        customized_collate_fn = partial(
            ux.custom_collate_fn,
            device=cfg.DEVICE,
            allowed_max_length=cfg.ALLOWED_MAX_LEN,
        )
        train_dl, val_dl, test_dl = ux.get_dataloaders(
            data=data, collate_fn=customized_collate_fn
        )

        run_finetuning_model(train_dl, val_dl, test_data, val_data)

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Finetuning completed in {execution_time_minutes:.2f} minutes.")
        print("DONE")
    elif cfg.MODE == "inference":
        # input_text = ux.format_input(cfg.user_input)
        input_text = cfg.user_input
        print(input_text)

        token_ids = generate(
            model=mx.get_finetuned_model(),
            idx=text_to_token_ids(input_text, cfg.tokenizer).to(cfg.DEVICE),
            max_new_tokens=cfg.MAX_NEW_GENERATED_TOKENS,
            context_size=cfg.BASE_MODEL_CONFIG["context_length"],
            eos_id=cfg.MODEL_EOT_ID,
        )
        generated_text = token_ids_to_text(token_ids, cfg.tokenizer)
        response_text = (
            generated_text[len(input_text) :].replace("### Response:", "").strip()
        )

        print("----" * 30)
        print(response_text)
        print("----" * 30)
