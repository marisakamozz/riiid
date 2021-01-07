import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import hydra
from hydra.utils import get_original_cwd
from pytorch_lightning import seed_everything

from util import get_path
from data import RiiidEmulator
from submit import make_submission
from saintmodel import *
from saintsubmit import load_saint_config, SaintPredictor


def train_model(args, run_name, history, test=None, saint_history=None):
    train, valid = train_test_split(history, test_size=0.01, random_state=args.seed)
    train_set = SaintDataset(train, seq_len=args.seq_len)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
    )
    valid_set = SaintDataset(valid, seq_len=args.seq_len)
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
    )

    logging.info('start train')
    model = SaintModel(
        seq_len=args.seq_len, n_dim=args.n_dim, std=args.std, dropout=args.dropout, nhead=args.nhead, n_layers=args.n_layers
    )
    module = SaintLightningModule(args, model)
    trainer = SaintTrainer(run_name, args)
    trainer.fit(module, train_loader, valid_loader)
    trainer.logger.experiment.log_artifact(trainer.logger.run_id, trainer.checkpoint.best_model_path)

    if test is not None:
        logging.info('start predict')
        module.load_state_dict(torch.load(trainer.checkpoint.best_model_path)['state_dict'])
        predictor = SaintPredictor(module.model, saint_history)
        env = RiiidEmulator(test)
        make_submission(env, predictor)
        score = env.score()
        trainer.logger.experiment.log_metric(trainer.logger.run_id, 'auc', score)


def run_cv(args):
    cv = args.cv
    run_name = f'CV{cv+1}'
    history = pd.read_pickle(get_path(f'../data/saint/cv{cv+1}_history.pickle'))
    # history = history[-10000:]
    valid = pd.read_pickle(get_path(f'../data/cvfiles/cv{cv+1}_valid.pickle'))
    valid = valid[:25000]
    last_history = pd.read_pickle(get_path(f'../data/saint/cv{cv+1}_last_history.pickle'))
    last_timestamp = pd.read_pickle(get_path(f'../data/saint/cv{cv+1}_last_timestamp.pickle'))
    dict_lag = pd.read_pickle(get_path(f'../data/saint/dict_lag.pickle'))
    dict_elapsed = pd.read_pickle(get_path(f'../data/saint/dict_elapsed.pickle'))
    saint_history = SaintHistory(args, last_history, last_timestamp, dict_lag, dict_elapsed)
    train_model(args, run_name, history, test=valid, saint_history=saint_history)


def run_submit(args):
    run_name = 'Submit'
    history = pd.read_pickle(get_path('../data/saint/history.pickle'))
    train_model(args, run_name, history)


@hydra.main(config_name='saint')
def main(args):
    logging.info('start')
    os.chdir(get_original_cwd())
    seed_everything(args.seed)
    try:
        if args.cv is None:
            run_submit(args)
        else:
            run_cv(args)
    except:
        logging.exception('exception')
    finally:
        logging.info('end')


if __name__ == "__main__":
    main()
