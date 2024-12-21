from transformers import BertModel, BertTokenizer
from transformers import BertConfig
import torch 
from torch import nn
from torch.nn import CrossEntropyLoss
import pennylane as qml
from lightning import LightningDataModule
# conver the dataset into pytorch_lighting datamudule
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
import logging
import json
from tempfile import TemporaryDirectory
import numpy as np 
from pathlib import Path
from collections import namedtuple
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse
import os

from transformers import (
    TextDatasetForNextSentencePrediction,
    DataCollatorForLanguageModeling
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")


def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    if len(tokens) > max_seq_length:
        logger.info('len(tokens): {}'.format(len(tokens)))
        logger.info('tokens: {}'.format(tokens))
        tokens = tokens[:max_seq_length]

    if len(tokens) != len(segment_ids):
        logger.info('tokens: {}\nsegment_ids: {}'.format(tokens, segment_ids))
        segment_ids = [0] * len(tokens)

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = int(epoch % num_data_epochs)
        logger.info('training_path: {}'.format(training_path))
        data_file = Path(training_path) / "epoch_{}.json".format(self.data_epoch)
        metrics_file = Path(training_path) / "epoch_{}_metrics.json".format(self.data_epoch)
        
        logger.info('data_file: {}'.format(data_file))
        logger.info('metrics_file: {}'.format(metrics_file))

        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path('./cache')
            input_ids = np.memmap(filename=self.working_dir / 'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir / 'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=bool)
            segment_ids = np.memmap(filename=self.working_dir / 'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=bool)
            lm_label_ids = np.memmap(filename=self.working_dir / 'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir / 'is_nexts.memmap', shape=(num_samples,), mode='w+', dtype=bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=bool)

        logging.info("Loading training examples for epoch {}".format(epoch))

        with data_file.open(encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                try:
                    example = json.loads(line)
                except Exception as e:
                    print("exception: {}".format(str(e)))
                    print("format error in line: {}".format(line))
                    # continue

                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next

        # assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(int(self.is_nexts[item])))
    
class QMLBertDataModule2(LightningDataModule):
    def __init__(
            self, 
            training_path: str, 
            tokenizer_path: str, 
            batch_size: int=32, 
            num_workers: int=1, 
            num_data_epochs: int=1, 
            mlm_probability: float=0.15, 
            max_seq_len: int=256,
            # reduce_memory: bool=False
        ):
        super().__init__()
        self.training_path = training_path
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_data_epochs = num_data_epochs
        self.mlm_probability = mlm_probability
        self.max_seq_len = max_seq_len
        
        self.save_hyperparameters(logger=False)

    def setup(self, stage=None):
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path, do_lower_case=True)

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True,
            mlm_probability=self.mlm_probability
        )

    def train_dataloader(self):
        data_epoch_id = self.trainer.current_epoch % self.num_data_epochs
        self.train_dataset = TextDatasetForNextSentencePrediction(tokenizer=self.tokenizer,
                                                                  file_path=os.path.join(self.training_path, f'epoch_{data_epoch_id}.txt'),
                                                                  block_size=self.max_seq_len)

        data_loader = DataLoader(self.train_dataset, 
                                 shuffle=True, 
                                 batch_size=self.batch_size, 
                                 collate_fn=self.data_collator,
                                 num_workers=self.num_workers,
                                 pin_memory=True)
        
        return data_loader
       
class QMLBertDataModule(LightningDataModule):
    def __init__(
            self, 
            training_path: str, 
            tokenizer_path: str, 
            batch_size: int=32, 
            num_workers: int=1, 
            num_data_epochs: int=1, 
            reduce_memory: bool=False
        ):
        super().__init__()
        self.training_path = training_path
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_data_epochs = num_data_epochs
        self.reduce_memory = reduce_memory
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
        # download data, pre-process, split, save to disk, etc...
        pass

    def setup(self, stage=None):
        # things to do on every process in DDP
        # load data, set variables, etc...
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path, do_lower_case=True)

    def train_dataloader(self):
        print('current training epoch : {}'.format(self.trainer.current_epoch))
        data_epoch_id = self.trainer.current_epoch % self.num_data_epochs
        self.train_dataset = PregeneratedDataset(self.training_path, data_epoch_id, self.tokenizer, self.num_data_epochs, self.reduce_memory)
  
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
