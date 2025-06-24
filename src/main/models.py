import torch

from src.config import config as cfg
from src.main.model_utils import GPTModel, load_weights_into_gpt
from src.main.pretrained_gpt_utils import download_and_load_gpt2


def get_pretrained_model():
    _, params = download_and_load_gpt2(
        model_size=cfg.MODEL_SIZE, models_dir=cfg.PRETRAINED_MODELS_DIR
    )
    model = GPTModel(cfg.BASE_MODEL_CONFIG)
    load_weights_into_gpt(model, params)
    model.to(cfg.DEVICE).eval()
    return model


def get_finetuned_model():
    model = GPTModel(cfg.BASE_MODEL_CONFIG)
    checkpoint = torch.load(cfg.FINETUNED_MODEL_FPATH, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint)
    model.to(cfg.DEVICE)
    model.eval()
    return model
