from transformers import BertModel, BertTokenizer
from transformers import BertConfig
from transformers import AdamW, AutoModel, get_linear_schedule_with_warmup
import torch 
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import pennylane as qml
import math
import sys
from torch import nn
from torch.nn import CrossEntropyLoss
from qml_bert_heads import BertPreTrainingHeads4QML, BertPreTrainingHeads

# import pytorch_lightning as pl

class QMLBertLitModule(LightningModule):
            
    def __init__(
        self,  
        bert_config: str,
        quantum_config: str = None,
        encoder_trainable: float = True,
        pretrained_encoder: bool = False,
        pretrain_head: str = 'quantum',
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
    ):
        #super().__init__()
        super(QMLBertLitModule, self).__init__()
        self.save_hyperparameters(logger=True)
        if pretrained_encoder:
            self.bert = BertModel.from_pretrained(bert_config,add_pooling_layer=False)
        else:
            self.bert_config = BertConfig.from_json_file(bert_config)
            self.bert = BertModel(self.bert_config, add_pooling_layer=False)
        if quantum_config is None:
            quantum_config = bert_config
            
        self.config = BertConfig.from_json_file(quantum_config)

        if pretrain_head == 'quantum':
            self.cls = BertPreTrainingHeads4QML(
                self.config, self.bert.embeddings.word_embeddings.weight)
            
        elif pretrain_head == 'classical':
            self.cls = BertPreTrainingHeads(
                self.config, self.bert.embeddings.word_embeddings.weight)

        # Freeze pre-trained encoder when asked
        for param in self.bert.parameters():
            param.requires_grad = False
        if encoder_trainable:
            for param in self.bert.parameters():
                param.requires_grad = True
        for name, param in self.bert.named_parameters():
            print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, next_sentence_label=None, position_ids=None):

        if  position_ids is not None and torch.sum(position_ids) == 0:
            position_ids = None
        sequence_output = self.bert(
                                    input_ids, 
                                    attention_mask, 
                                    token_type_ids, 
                                    position_ids=position_ids,
                                    output_attentions=False)[:1][0]
        
        
        prediction_scores, seq_relationship_score = self.cls(sequence_output)

        #if masked_lm_labels is not None and next_sentence_label is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
        
        total_loss = masked_lm_loss + next_sentence_loss
        nsp_acc = (seq_relationship_score.argmax(dim=-1) == next_sentence_label).float().mean()

        return total_loss, nsp_acc, masked_lm_loss, next_sentence_loss
        
    def get_total_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices=1
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        print(f"{self.hparams.learning_rate =}")
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.get_total_training_steps(),
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    

    
    def training_step(self, batch, batch_idx):
        input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
        loss, nsp_acc, mlm_loss, nsp_loss = self(input_ids, segment_ids, input_mask, lm_label_ids, is_next)

        self.log('train_loss', loss.item())
        self.log('train_nsp_acc', nsp_acc)
        self.log('train_masked_lm_loss', mlm_loss)
        self.log('train_next_sentence_loss', nsp_loss)
        return loss