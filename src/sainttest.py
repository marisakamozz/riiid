import time
import logging
import pandas as pd
import torch
from pytorch_lightning import seed_everything

from util import init, get_path
from data import SEED, RiiidEmulator
from submit import make_submission
from saintmodel import *
from saintsubmit import load_saint_config
from saintsubmit import SaintPredictor


def main(run_name):
    seed_everything(SEED)
    args = load_saint_config()
    model = SaintModel(
        seq_len=args.seq_len, n_dim=args.n_dim, std=args.std, dropout=args.dropout, nhead=args.nhead, n_layers=args.n_layers
    )
    module = SaintLightningModule(args, model)
    module.load_state_dict(torch.load(get_path('saint.ckpt'))['state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module.to(device)

    cv = 4
    last_history = pd.read_pickle(get_path(f'../data/saint/cv{cv+1}_last_history.pickle'))
    last_timestamp = pd.read_pickle(get_path(f'../data/saint/cv{cv+1}_last_timestamp.pickle'))
    last_user_count = pd.read_pickle(get_path(f'../data/saint/cv{cv+1}_last_user_count.pickle'))
    dict_lag = pd.read_pickle(get_path(f'../data/saint/dict_lag.pickle'))
    dict_elapsed = pd.read_pickle(get_path(f'../data/saint/dict_elapsed.pickle'))
    dict_user_count = pd.read_pickle(get_path('../data/saint/dict_user_count.pickle'))
    saint_history = SaintHistory(args, last_history, last_timestamp, last_user_count, dict_lag, dict_user_count, dict_elapsed)
    predictor = SaintPredictor(module.model, saint_history)

    valid = pd.read_pickle(get_path(f'../data/cvfiles/cv{cv+1}_valid.pickle'))
    valid = valid[:50000]
    env = RiiidEmulator(valid)

    start = time.time()
    make_submission(env, predictor)
    elapsed_time = time.time() - start
    logging.info(f'{elapsed_time:.3f}[sec]')
    logging.info(f'{env.score():.5f}[AUC]')


if __name__ == "__main__":
    run_name = init(__file__)
    try:
        main(run_name)
    except:
        logging.exception('exception')
    finally:
        logging.info('end')
