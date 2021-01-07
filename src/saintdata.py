import gc
import logging
import numpy as np
import pandas as pd

from util import init, get_path
from data import read_train
from sainttrain import load_saint_config


QUANTILES = np.arange(0, 1, 0.01)


def user_count_interval():
    left = [0, 3, 4, 7, 10, 13, 16, 22, 26, 30]
    right = left[1:] + [np.iinfo(np.int32).max]
    interval_user_count = pd.IntervalIndex.from_arrays(left, right)
    dict_user_count = pd.Series(range(len(interval_user_count)), index=interval_user_count.values)
    return dict_user_count


def make_history(train, seq_len, interval_lag=None, interval_elapsed=None):
    logging.info('last timestamp')
    last_timestamp = train[['user_id', 'timestamp']].groupby('user_id').tail(1)
    last_timestamp = last_timestamp.set_index('user_id')['timestamp']

    logging.info('categorize lag')
    train['lag'] = train[['user_id', 'timestamp']].groupby('user_id').diff()
    if interval_lag is None:
        right = train.lag.quantile(QUANTILES).unique()
        right = np.insert(right, 0, -1.0)
        right = np.append(right[:-1], np.finfo(np.float32).max)
        left = np.insert(right[:-1], 0, -2.0)
        interval_lag = pd.IntervalIndex.from_arrays(left, right)
    train.lag.fillna(-1, inplace=True)
    cut = pd.cut(train.lag, bins=interval_lag)
    dict_lag = pd.Series(range(len(interval_lag)), index=interval_lag.values)
    train['lag'] = dict_lag.reindex(cut).values

    train = train[train.content_type_id == 0].copy()

    logging.info('categorize elapsed')
    train['elapsed'] = train[['user_id', 'prior_question_elapsed_time']].groupby('user_id').shift(-1)
    if interval_elapsed is None:
        right = train.elapsed.quantile(QUANTILES).unique()
        right = np.insert(right, 0, -1.0)
        right = np.append(right[:-1], np.finfo(np.float32).max)
        left = np.insert(right[:-1], 0, -2.0)
        interval_elapsed = pd.IntervalIndex.from_arrays(left, right)
    train.elapsed.fillna(-1, inplace=True)
    cut = pd.cut(train.elapsed, bins=interval_elapsed)
    dict_elapsed = pd.Series(range(len(interval_elapsed)), index=interval_elapsed.values)
    train['elapsed'] = dict_elapsed.reindex(cut).values

    logging.info('categorize explained')
    train['explained'] = train[['user_id', 'prior_question_had_explanation']].groupby('user_id').shift(-1)
    train.explained.fillna(2, inplace=True)
    train['explained'] = train.explained.astype('int8')

    logging.info('categorize user_count')
    dict_user_count = user_count_interval()
    train['user_count'] = train[['user_id', 'row_id']].groupby('user_id').cumcount() + 1
    train['user_count'] = dict_user_count.reindex(train.user_count).values
    last_user_count = train[['user_id', 'row_id']].groupby('user_id').count()['row_id'] + 1

    train = train[['user_id', 'timestamp', 'content_id', 'lag', 'user_count', 'answered_correctly', 'elapsed', 'explained']].copy()
    train = train.set_index(['user_id', 'timestamp']).sort_index().reset_index().set_index('user_id')

    def to_history(df):
        return (
            df[['content_id', 'lag', 'user_count']].values,
            df[['answered_correctly', 'elapsed', 'explained']].values
        )
    
    logging.info('groupby to make history')
    history = train.groupby(level=0).apply(to_history)

    def to_last_history(df):
        return (
            df[['content_id', 'lag', 'user_count']].tail(seq_len).values,
            df[['answered_correctly', 'elapsed', 'explained']].tail(seq_len).values
        )
    
    logging.info('groupby to make last history')
    last_history = train.groupby(level=0).apply(to_last_history)

    return history, last_history, last_timestamp, last_user_count, dict_lag, dict_elapsed, dict_user_count, interval_lag, interval_elapsed


def main(run_name):
    args = load_saint_config()
    seq_len = args.seq_len

    train = read_train()
    history, last_history, last_timestamp, last_user_count, dict_lag, dict_elapsed, dict_user_count, interval_lag, interval_elapsed = make_history(train, seq_len)
    logging.info('make history end')
    pd.to_pickle(history, get_path('../data/saint/history.pickle'))
    pd.to_pickle(last_history, get_path('../data/saint/last_history.pickle'))
    pd.to_pickle(last_timestamp, get_path('../data/saint/last_timestamp.pickle'))
    pd.to_pickle(last_user_count, get_path('../data/saint/last_user_count.pickle'))
    pd.to_pickle(dict_lag, get_path('../data/saint/dict_lag.pickle'))
    pd.to_pickle(dict_elapsed, get_path('../data/saint/dict_elapsed.pickle'))
    pd.to_pickle(dict_user_count, get_path('../data/saint/dict_user_count.pickle'))
    logging.info('save history end')
    del train, history, last_history, last_timestamp, last_user_count, dict_lag, dict_elapsed, dict_user_count
    gc.collect()

    for cv in range(5):
        train = pd.read_pickle(get_path(f'../data/cvfiles/cv{cv+1}_train.pickle'))
        history, last_history, last_timestamp, last_user_count, _, _, _, _, _ = make_history(train, seq_len, interval_lag, interval_elapsed)
        logging.info(f'[CV{cv+1}] make history end')
        pd.to_pickle(history, get_path(f'../data/saint/cv{cv+1}_history.pickle'))
        pd.to_pickle(last_history, get_path(f'../data/saint/cv{cv+1}_last_history.pickle'))
        pd.to_pickle(last_timestamp, get_path(f'../data/saint/cv{cv+1}_last_timestamp.pickle'))
        pd.to_pickle(last_user_count, get_path(f'../data/saint/cv{cv+1}_last_user_count.pickle'))
        logging.info(f'[CV{cv+1}] save history end')
        del train, history, last_history, last_timestamp, last_user_count
        gc.collect()


if __name__ == "__main__":
    run_name = init(__file__)
    try:
        main(run_name)
    except:
        logging.exception('exception')
    finally:
        logging.info('end')
