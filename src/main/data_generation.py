from src.config import config as cfg
from src.main.data_utils import (
    generate_dataset_proportional,
    plot_label_distribution,
    save_json,
)

if __name__ == "__main__":
    rules_path = cfg.RULES_FPATH
    dataset = generate_dataset_proportional(
        rules_path, n=cfg.NUM_SYNTHETIC_DATA, proportions=cfg.SYNTHETIC_DATA_PROPORTIONS
    )
    save_json(dataset, cfg.DATASET_FPATH)
    # hf_dataset = to_hf_dataset(dataset)
    # plot_label_distribution(dataset)
