import json
import sys
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder import CTCTextEncoder


def main():
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    if len(sys.argv) != 3:
        raise AssertionError(
            "Exactly 2 arguments must be provided: path to predictions and path to ground truth"
        )

    preds_dir, ground_truth_dir = sys.argv[1], sys.argv[2]

    cers, wers = [], []

    for preds_path, ground_truth_path in zip(
        sorted(Path(preds_dir).iterdir()), sorted(Path(ground_truth_dir).iterdir())
    ):
        with preds_path.open() as pred_f:
            pred = CTCTextEncoder.normalize_text(json.load(pred_f)["pred_text"])

        with ground_truth_path.open() as gt_f:
            ground_truth = CTCTextEncoder.normalize_text(gt_f.readline())

        cers.append(calc_cer(ground_truth, pred))
        wers.append(calc_wer(ground_truth, pred))

    print(f"Avg CER: {sum(cers) / len(cers):.4f}")
    print(f"Avg WER: {sum(wers) / len(wers):.4f}")


if __name__ == "__main__":
    main()
