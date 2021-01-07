import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import roc_auc_score

from data import TARGET

ACCEPTED_USER_CONTENT_SIZE = 3


class SaintDataset(torch.utils.data.Dataset):
    def __init__(self, history, seq_len, step_size=75):
        super().__init__()
        self.history = history
        self.seq_len = seq_len
        self.indexes = []

        for user_id in history.index:
            q, _ = history[user_id]
            if len(q) < ACCEPTED_USER_CONTENT_SIZE:
                continue

            if len(q) <= self.seq_len:
                self.indexes.append((user_id, 0, len(q), 0))
                continue

            start = 0
            end = self.seq_len
            mask_index = 0
            self.indexes.append((user_id, start, end, mask_index))

            mask_index = end
            loop_count = (len(q) - self.seq_len - 1) // step_size + 1
            for i in range(loop_count):
                start = (i + 1) * step_size
                end = start + self.seq_len
                if end > len(q):
                    start -= (end - len(q))
                    end = len(q)
                self.indexes.append((user_id, start, end, mask_index))
                mask_index = end

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        user_id, start, end, mask_index = self.indexes[index]
        q, qa = self.history[user_id]

        x1 = np.zeros((self.seq_len, 4), dtype=np.int32)
        x2 = np.zeros((self.seq_len, 4), dtype=np.int32)
        y  = np.zeros((self.seq_len, 1), dtype=np.int32)
        weight = np.zeros((self.seq_len, 1), dtype=np.float32)

        length = end - start
        weight_length = end - mask_index
        position = np.array(range(1, length+1), dtype=np.int32)
        if length < self.seq_len:
            x1[-length  :, 0]   = position
            x1[-length  :, 1:4] = q
            x2[-length+1:, 0]   = position[:-1]
            x2[-length+1:, 1:4] = qa[:-1]
            y[-length  :]       = qa[:, 0:1]
            weight[-weight_length:] = np.ones((weight_length, 1), dtype=np.float32)
        else:
            assert length == self.seq_len
            x1[ :, 0]   = position
            x1[ :, 1:4] = q[start:end]
            x2[1:, 0]   = position[1:]
            x2[1:, 1:4] = qa[start:end-1]
            y[  :]      = qa[start:end, 0:1]
            weight[-weight_length:] = np.ones((weight_length, 1), dtype=np.float32)
        
        return x1, x2, y, weight


class SaintTestDataset(torch.utils.data.Dataset):
    def __init__(
        self, test, last_history, last_timestamp, last_user_count,
        dict_lag, dict_user_count, dict_elapsed,
        seq_len, position
    ):
        super().__init__()
        self.test = test
        self.last_history = last_history
        self.last_timestamp = last_timestamp
        self.last_user_count = last_user_count
        self.dict_lag = dict_lag
        self.dict_user_count = dict_user_count
        self.dict_elapsed = dict_elapsed
        self.seq_len = seq_len
        self.position = position

    def __len__(self):
        return len(self.test)

    def __getitem__(self, index):
        row = self.test.iloc[index]
        user_id = row['user_id']
        question_id = row['content_id']
        x1 = np.zeros((self.seq_len, 4), dtype=np.int32)
        x2 = np.zeros((self.seq_len, 4), dtype=np.int32)

        if user_id not in self.last_history.index:
            x1[-1, 1] = question_id
            x1[-1, 2] = 0
            x1[-1, 3] = 0
            return x1, x2

        q, qa = self.last_history[user_id]

        if len(q) >= self.seq_len - 1:
            history_len = self.seq_len - 1
        else:
            history_len = len(q)
        
        position = self.position[:history_len+1]
        last_timestamp = self.last_timestamp[user_id]
        lag = row['timestamp'] - last_timestamp
        lag = self.dict_lag[lag]
        user_count = self.dict_user_count[self.last_user_count[user_id]]
        elapsed = row['prior_question_elapsed_time']
        elapsed = self.dict_elapsed[elapsed]
        explained = int(row['prior_question_had_explanation'])

        if len(q) >= self.seq_len - 1:
            x1[: , 0]   = position
            x2[1:, 0]   = position[:-1]
            x1[:-1, 1:4] = q[-history_len:]
            x2[1: , 1:4] = qa[-history_len:]
        else:
            x1[-history_len-1:  , 0]   = position
            x2[-history_len:    , 0]   = position[:-1]
            x1[-history_len-1:-1, 1:4] = q
            x2[-history_len:    , 1:4] = qa
        x1[-1, 1] = question_id
        x1[-1, 2] = lag
        x1[-1, 3] = user_count
        x2[-1, 2] = elapsed
        x2[-1, 3] = explained
        return x1, x2


class SaintHistory():
    def __init__(self, args, last_history, last_timestamp, last_user_count, dict_lag, dict_user_count, dict_elapsed):
        self.last_history = last_history
        self.last_timestamp = last_timestamp
        self.last_user_count = last_user_count
        self.dict_lag = dict_lag
        self.dict_user_count = dict_user_count
        self.dict_elapsed = dict_elapsed
        self.seq_len = args.seq_len
        self.batch_size = args.batch_size
        self.position = np.array(range(1, self.seq_len+1))

    def make_data_loader(self, test):
        dataset = SaintTestDataset(
            test, self.last_history, self.last_timestamp, self.last_user_count,
            self.dict_lag, self.dict_user_count, self.dict_elapsed,
            self.seq_len, self.position
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(test),
            shuffle=False,
            num_workers=os.cpu_count(),
            # pin_memory=True,
        )
        return dataloader

    def update(self, test):
        for i, row in test.iterrows():
            self._update_history(row)

    def _update_history(self, row):
        timestamp = row['timestamp']
        user_id = row['user_id']
        question_id = row['content_id']
        content_type_id = row['content_type_id']
        correct = row[TARGET]

        if user_id not in self.last_history.index:
            if content_type_id == 1:
                return
            q = np.array([[question_id, 0, 0]])
            qa = np.array([[correct, 0, 2]])
            self.last_history[user_id] = (q, qa)
            self.last_timestamp[user_id] = timestamp
            self.last_user_count[user_id] = 2
            return
        
        if content_type_id == 1:
            self.last_timestamp[user_id] = timestamp
            return

        q, qa = self.last_history[user_id]

        lag = timestamp - self.last_timestamp[user_id]
        lag = self.dict_lag[lag]
        user_count = self.dict_user_count[self.last_user_count[user_id]]
        new_q = np.array([[question_id, lag, user_count]])

        qa[-1, 1] = self.dict_elapsed[row['prior_question_elapsed_time']]
        qa[-1, 2] = int(row['prior_question_had_explanation'])
        new_qa = np.array([[correct, 0, 2]])

        q = np.concatenate([q, new_q], axis=0)
        qa = np.concatenate([qa, new_qa], axis=0)
        self.last_history[user_id] = (q, qa)
        self.last_timestamp[user_id] = timestamp
        self.last_user_count[user_id] += 1


class SaintModel(torch.nn.Module):
    def __init__(
        self, seq_len, n_dim=64, std=0.01, dropout=0.1, n_question=13523, n_lag=79, n_count=10, n_elapsed=65, nhead=8, n_layers=2
    ):
        super().__init__()
        self.position_emb  = torch.nn.Embedding(seq_len+1 , n_dim)
        self.question_emb  = torch.nn.Embedding(n_question, n_dim)
        self.lag_emb       = torch.nn.Embedding(n_lag     , n_dim)
        self.count_emb     = torch.nn.Embedding(n_count   , n_dim)
        self.correct_emb   = torch.nn.Embedding(2         , n_dim)
        self.elapsed_emb   = torch.nn.Embedding(n_elapsed , n_dim)
        self.explained_emb = torch.nn.Embedding(2         , n_dim)
        torch.nn.init.normal_(self.position_emb.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.question_emb.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.lag_emb.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.count_emb.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.correct_emb.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.elapsed_emb.weight, mean=0.0, std=std)
        torch.nn.init.normal_(self.explained_emb.weight, mean=0.0, std=std)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            n_dim,
            nhead=nhead,
            dim_feedforward=n_dim,
            dropout=dropout,
        )
        layer_norm = torch.nn.LayerNorm(n_dim)
        self.predecoder = torch.nn.TransformerEncoder(encoder_layer, n_layers, layer_norm)
        self.transformer = torch.nn.Transformer(
            n_dim,
            nhead=nhead,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=n_dim,
            dropout=dropout,
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(n_dim, n_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(n_dim, n_dim),
            torch.nn.Dropout(p=dropout),
        )
        self.layer_norm = torch.nn.LayerNorm(n_dim)
        self.linear = torch.nn.Linear(n_dim, 1)
        self.mask = self.generate_square_subsequent_mask(seq_len)
    
    def forward(self, x1, x2):
        mask = self.mask.to(x1.device)
        x1, x2 = x1.long(), x2.long()
        # x1 : batch_size * seq_len * 4
        p1 = self.position_emb(x1[:, :, 0])
        q = self.question_emb(x1[:, :, 1])
        l = self.lag_emb(x1[:, :, 2])
        n = self.count_emb(x1[:, :, 3])
        # x2 : batch_size * seq_len * 4
        p2 = self.position_emb(x2[:, :, 0])
        c = self.correct_emb(x2[:, :, 1])
        e = self.elapsed_emb(x2[:, :, 2])
        x = self.explained_emb(x2[:, :, 3])
        # p1, q, l, n, p2, c, e, x : batch_size * seq_len * n_dim
        x1 = p1 + q + l + n
        x2 = p2 + c + e + x
        # x1, x2 : batch_size * seq_len * n_dim
        x1, x2 = x1.transpose(0, 1), x2.transpose(0, 1)
        # x1, x2 : seq_len * batch_size * n_dim
        x2 = self.predecoder(x2, mask=mask)
        y = self.transformer(x1, x2, src_mask=mask, tgt_mask=mask)
        # y : seq_len * batch_size * n_dim
        y = y.transpose(0, 1)
        # y : batch_size * seq_len * n_dim
        y = self.ffn(y) + y
        # y : batch_size * seq_len * n_dim
        y = self.linear(self.layer_norm(y))
        # y : batch_size * seq_len * 1
        return y
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class SaintLightningModule(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.hparams = {key:value for key, value in args.items()}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x1, x2, y, w = batch
        y_pred = self.model(x1, x2).view(-1)
        y = y.float().view(-1)
        w = w.float().view(-1)
        criterion = torch.nn.BCEWithLogitsLoss(w, reduction='sum')
        loss = criterion(y_pred, y) / w.sum()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y, w = batch
        y_pred = self.model(x1, x2)
        y_pred = y_pred[:, -1].view(-1, 1)
        y = y[:, -1].float().view(-1, 1)
        w = w[:, -1].float().view(-1, 1)
        return torch.cat([y_pred, y, w], dim=1)

    def validation_epoch_end(self, valid_step_outputs):
        results = torch.cat(valid_step_outputs, dim=0)
        y_pred = results[:, 0]
        y      = results[:, 1]
        weight = results[:, 2]
        if y.sum() > 0:
            criterion = torch.nn.BCEWithLogitsLoss(weight, reduction='sum')
            loss = criterion(y_pred, y) / weight.sum()
            self.log('val_loss', loss)
            results = pd.DataFrame(results.cpu().numpy(), columns=['y_pred', 'y', 'weight'])
            results = results[results.weight == 1]
            score = roc_auc_score(results.y, results.y_pred)
            self.log('val_auc', score)
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


class SaintTrainer(pl.Trainer):
    def __init__(self, run_name, args):
        if args.patience is None:
            callbacks = None
        else:
            self.checkpoint = ModelCheckpoint(
                monitor='val_auc',
                dirpath='../models/',
                filename=f'SAINT-{run_name}',
                mode='max'
            )
            early_stopping = EarlyStopping(
                monitor='val_auc',
                patience=args.patience,
                verbose=False,
                mode='max'
            )
            callbacks = [self.checkpoint, early_stopping]
        tags = {'mlflow.runName': run_name}
        logger = MLFlowLogger(args.experiment, 'file:/kaggle/logs/mlruns', tags)
        super().__init__(
            gpus=1,
            deterministic=True,
            max_epochs=args.max_epochs,
            logger=logger,
            val_check_interval=0.2,
            callbacks=callbacks,
        )
