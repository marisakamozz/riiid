from os import path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from util import get_path

SEED = 1234

TARGET = 'answered_correctly'

DTYPES_TRAIN = {
    'row_id': 'int64',
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'task_container_id': 'int16',
    'user_answer': 'int8',
    'answered_correctly':'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'boolean'
}

def read_train(usecols=None):
    cache_path = get_path('../data/train.pickle')
    if path.exists(cache_path):
        if usecols is None:
            return pd.read_pickle(cache_path)
        else:
            return pd.read_pickle(cache_path)[usecols]
    train_path = get_path('../input/riiid-test-answer-prediction/train.csv')
    if usecols is None:
        train = pd.read_csv(train_path, dtype=DTYPES_TRAIN)
        train.to_pickle(cache_path)
        return train
    else:
        return pd.read_csv(train_path, dtype=DTYPES_TRAIN, usecols=usecols)

DTYPES_QUESTIONS = {
    'question_id': 'int16',
    'bundle_id': 'int16',
    'correct_answer': 'int8',
    'part': 'int8',
    'tags': 'object',
}
TAG_COLUMNS = [f'tag{i+1}' for i in range(6)]
CAT_TAGS = pd.CategoricalDtype(list(range(188)))

def read_questions():
    questions = pd.read_csv(get_path('../input/riiid-test-answer-prediction/questions.csv'), dtype=DTYPES_QUESTIONS)
    tags = questions.tags.apply(lambda x: pd.Series(x.split(), dtype=float) if type(x) == str else pd.Series(dtype=float))
    tags.columns = TAG_COLUMNS
    tags = tags.astype(CAT_TAGS)
    questions = pd.concat([questions, tags], axis=1)
    questions.set_index('question_id', inplace=True)
    return questions


class RiiidEmulator():
    def __init__(self, df, max_user=1000):
        self.df = df
        self.max_user = max_user
        self.y_pred = None

    def iter_test(self):
        self.current_index = 0
        self.progress_bar = tqdm(total=len(self.df))
        self.iterator = RiiidIterator(self.df, self.max_user)
        return self.iterator

    def predict(self, y_pred):
        self.progress_bar.update(self.iterator.current - self.current_index)
        self.current_index = self.iterator.current
        if self.y_pred is None:
            self.y_pred = y_pred
        else:
            self.y_pred = pd.concat([self.y_pred, y_pred])

    def score(self):
        y_pred = self.y_pred.copy()
        y_pred.columns = ['row_id', 'y_pred']
        y_true = self.df[self.df.content_type_id == 0][['row_id', TARGET]].copy()
        y_all = y_true.merge(y_pred, on='row_id')
        assert len(y_true) == len(y_all)
        return roc_auc_score(y_all[TARGET], y_all['y_pred'])


# "Time-series API (iter_test) Emulator" by tito
# https://www.kaggle.com/its7171/time-series-api-iter-test-emulator
class RiiidIterator(object):
    def __init__(self, df, max_user=1000):
        df = df.reset_index(drop=True)
        self.df = df
        self.user_answer = df['user_answer'].astype(str).values
        self.answered_correctly = df[TARGET].astype(str).values
        df['prior_group_responses'] = "[]"
        df['prior_group_answers_correct'] = "[]"
        self.sample_df = df[df['content_type_id'] == 0][['row_id']]
        self.sample_df[TARGET] = 0
        self.len = len(df)
        self.user_id = df.user_id.values
        self.task_container_id = df.task_container_id.values
        self.content_type_id = df.content_type_id.values
        self.max_user = max_user
        self.current = 0
        self.pre_user_answer_list = []
        self.pre_answered_correctly_list = []

    def __iter__(self):
        return self
    
    def fix_df(self, user_answer_list, answered_correctly_list, pre_start):
        df = self.df[pre_start:self.current].copy()
        sample_df = self.sample_df[pre_start:self.current].copy()
        df.loc[pre_start,'prior_group_responses'] = '[' + ",".join(self.pre_user_answer_list) + ']'
        df.loc[pre_start,'prior_group_answers_correct'] = '[' + ",".join(self.pre_answered_correctly_list) + ']'
        self.pre_user_answer_list = user_answer_list
        self.pre_answered_correctly_list = answered_correctly_list
        return df, sample_df

    def __next__(self):
        added_user = set()
        pre_start = self.current
        pre_added_user = -1
        pre_task_container_id = -1

        user_answer_list = []
        answered_correctly_list = []
        while self.current < self.len:
            crr_user_id = self.user_id[self.current]
            crr_task_container_id = self.task_container_id[self.current]
            crr_content_type_id = self.content_type_id[self.current]
            if crr_content_type_id == 1:
                # no more than one task_container_id of "questions" from any single user
                # so we only care for content_type_id == 0 to break loop
                user_answer_list.append(self.user_answer[self.current])
                answered_correctly_list.append(self.answered_correctly[self.current])
                self.current += 1
                continue
            if crr_user_id in added_user and ((crr_user_id != pre_added_user) or (crr_task_container_id != pre_task_container_id)):
                # known user(not prev user or differnt task container)
                return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
            if len(added_user) == self.max_user:
                if  crr_user_id == pre_added_user and crr_task_container_id == pre_task_container_id:
                    user_answer_list.append(self.user_answer[self.current])
                    answered_correctly_list.append(self.answered_correctly[self.current])
                    self.current += 1
                    continue
                else:
                    return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
            added_user.add(crr_user_id)
            pre_added_user = crr_user_id
            pre_task_container_id = crr_task_container_id
            user_answer_list.append(self.user_answer[self.current])
            answered_correctly_list.append(self.answered_correctly[self.current])
            self.current += 1
        if pre_start < self.current:
            return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
        else:
            raise StopIteration()
