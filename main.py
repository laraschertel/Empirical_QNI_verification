import argparse

from config import TrainConfig, ExperimentConfig
from experiments_fairness import run_fairness_experiment
from experiments_membership import run_membership_experiment
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["fairness", "membership"], default="membership")
    p.add_argument("--dataset", choices=["adult", "compas", "mnist"], default="mnist")
    p.add_argument("--protected_attr", choices=["gender", "race"], default="race")
    p.add_argument("--dp", action="store_true", default=False)
    return p.parse_args()



def main():
    args = parse_args()

    train_cfg = TrainConfig(
        dp=args.dp,
        device="cuda" if torch.cuda.is_available() else "cpu", 
        use_pretrained=True,
        save_model=True,
    )
    exp_cfg = ExperimentConfig(
        task=args.task,
        dataset=args.dataset,
    )

    if args.task == "fairness":
        run_fairness_experiment(train_cfg, exp_cfg)
    else:
        run_membership_experiment(train_cfg, exp_cfg)


if __name__ == "__main__":
    main()
