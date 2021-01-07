from os import path
import logging
import pathlib
import yaml
import numpy as np
import pandas as pd

def init(file_name):
    run_name = pathlib.Path(file_name).stem
    formatter = f'%(asctime)s %(levelname)s {run_name} %(message)s'
    logging.basicConfig(filename=get_path('../logs/messages.log'), format=formatter, level=logging.INFO)
    logging.info('start')
    return run_name

# "FAST PANDAS LEFT JOIN (357x faster than pd.merge)" by tkm2261
# https://www.kaggle.com/tkm2261/fast-pandas-left-join-357x-faster-than-pd-merge
def fast_merge(left, right, key):
    return pd.concat([left.reset_index(drop=True), right.reindex(left[key].values).reset_index(drop=True)], axis=1)

def get_path(filename):
    return path.join(path.dirname(__file__), filename)

# "Elo world" by FabienDaniel
# https://www.kaggle.com/fabiendaniel/elo-world
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

class AttributeDict(object):
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def fields(self):
        return self.obj

    def keys(self):
        return self.obj.keys()
    
    def items(self):
        return self.obj.items()
