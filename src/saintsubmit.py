import logging
import yaml
import torch
from pytorch_lightning import seed_everything

from util import init, get_path, AttributeDict
from data import TARGET
from saintmodel import *


def load_saint_config():
    with open(get_path('saint.yaml')) as file:
        config = yaml.safe_load(file)
    return AttributeDict(config)


class SaintPredictor():
    def __init__(self, model, saint_history):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.saint_history = saint_history

    def update_model(self, test):
        self.saint_history.update(test)

    def predict(self, test):
        test = test[test.content_type_id == 0]
        test_loader = self.saint_history.make_data_loader(test)
        with torch.no_grad():
            y_pred = []
            x1, x2 = next(iter(test_loader))
            x1, x2 = x1.to(self.device), x2.to(self.device)
            y_pred = torch.sigmoid(self.model(x1, x2)[:, -1, 0])
        return pd.DataFrame({'row_id': test.row_id, TARGET: y_pred.cpu().numpy()})


def make_submission(env, predictor):
    iteration = env.iter_test()
    previous_test_df = None
    for (current_test, sample_prediction_df) in iteration:
        if previous_test_df is not None:
            answers = eval(current_test["prior_group_answers_correct"].iloc[0])
            responses = eval(current_test["prior_group_responses"].iloc[0])
            previous_test_df[TARGET] = answers
            previous_test_df['user_answer'] = responses
            predictor.update_model(previous_test_df)
        
        previous_test_df = current_test.copy()
        env.predict(predictor.predict(current_test))


def main(run_name):
    logging.info('start')
    seed_everything(SEED)
    import sys
    sys.path.append('../input/riiid-test-answer-prediction/')
    args = load_saint_config()
    model = SaintModel(
        seq_len=args.seq_len, n_dim=args.n_dim, std=args.std, dropout=args.dropout, nhead=args.nhead, n_layers=args.n_layers
    )
    module = SaintLightningModule(args, model)
    module.load_state_dict(torch.load(get_path('saint.ckpt'))['state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module.to(device)

    last_history = pd.read_pickle(get_path('../data/saint/last_history.pickle'))
    last_timestamp = pd.read_pickle(get_path('../data/saint/last_timestamp.pickle'))
    last_user_count = pd.read_pickle(get_path('../data/saint/last_user_count.pickle'))
    dict_lag = pd.read_pickle(get_path('../data/saint/dict_lag.pickle'))
    dict_elapsed = pd.read_pickle(get_path('../data/saint/dict_elapsed.pickle'))
    dict_user_count = pd.read_pickle(get_path('../data/saint/dict_user_count.pickle'))
    saint_history = SaintHistory(args, last_history, last_timestamp, last_user_count, dict_lag, dict_user_count, dict_elapsed)
    predictor = SaintPredictor(module.model, saint_history)

    import riiideducation
    env = riiideducation.make_env()
    make_submission(env, predictor)


if __name__ == "__main__":
    run_name = init(__file__)
    try:
        main(run_name)
    except:
        logging.exception('exception')
    finally:
        logging.info('end')
