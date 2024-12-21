from transformers import BertConfig,BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import torch 
from typing import Optional
from collections import defaultdict
from datetime import datetime
from lightning import LightningModule
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from qml_bert_pretrain_module import QMLBertLitModule
import datasets
import math
import numpy as np
import pennylane as qml
import random
# import pytorch_lightning as pl
        
class GLUETransformerLitModule(LightningModule):

    task_info_map = {
        'cola':(2,['validation']),
        'mrpc':(2,['validation']),
        'sst2':(2,['validation']),
        'qqp':(2, ['validation']),
        'stsb':(1, ['validation']),
        'mnli':(3, ['validation_matched', 'validation_mismatched']),
        'wnli':(2, ['validation']),
        'ax':(3, []),
        'rte':(2,['validation']),
        'qnli':(2,['validation']),
        'mc':(2,['validation']),
        'rp':(2,['validation']),
        'iacv1':(2,['validation']),
        'iacv2':(2,['validation']),
        'OLID':(2,['validation']),
        'TwitterHate_fold0':(2,['validation']),
        'TwitterHate_fold1':(2,['validation']),
        'TwitterHate_fold2':(2,['validation']),
        'TwitterHate_fold3':(2,['validation']),
        'TwitterHate_fold4':(2,['validation'])
    }
    
    def get_circuit(self):
        dev = qml.device('default.qubit', wires=self.num_qubits)
        @qml.qnode(dev, interface='torch')
        def quantum_model(w, unitary_params):
            qml.templates.BasicEntanglerLayers(w, wires=range(self.num_qubits))
            qml.templates.StronglyEntanglingLayers(unitary_params, wires=range(self.num_qubits))
            for wire in range(self.num_qubits):
                if np.random.rand() < self.bitflip_prob:
                    qml.PauliX(wires=wire)  # bit flip
            return qml.probs(wires=range(self.num_qubits))
        return quantum_model
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        task_name: str,
        num_labels: int = 2,
        learning_rate: float = 3e-4,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        n_unitary_circuit_layers: int = 5,
        eval_splits: Optional[list] = None,
        pretrained: float = True,
        useQChead: float = False,
        encoder_trainable: float = False,
        projection_trainable: float = True,
        **kwargs,
    ):
        super().__init__()
        num_labels, eval_splits = self.task_info_map[task_name]
        self.save_hyperparameters()

        self.pretrained = pretrained
        self.useQChead = useQChead
        BertLayerNorm = torch.nn.LayerNorm

        self.config = BertConfig.from_json_file(config_path)

        self.bert = BertModel.from_pretrained('bert-base-uncased',add_pooling_layer=True, return_dict=False)

        for name, param in self.bert.named_parameters():
            if 'weight' in name:
                param.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif 'bias' in name:
                param.data.zero_()

        self.num_qubits = self.config.n_qubits
        self.num_unitary_circuit_layers = n_unitary_circuit_layers
        self.bitflip_prob = self.config.bitflip_prob
        
        
        self.model = QMLBertLitModule.load_from_checkpoint(checkpoint_path=checkpoint_path, 
                                                           bert_config=config_path, 
                                                           quantum_config=config_path,
                                                           pretrained_encoder=False)
        
                                               
        # Freeze pre-trained encoder when asked
        for param in self.model.bert.parameters():
            param.requires_grad = True

        if not encoder_trainable:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        self.num_labels = num_labels
        self.entangle_params_transform_layer = self.model.cls.entangle_params_transform_layer
        
        if not self.pretrained and not self.useQChead:
            input_size = self.model.cls.entangle_params_transform_layer.in_features
            output_size = self.model.cls.entangle_params_transform_layer.out_features
            self.entangle_params_transform_layer = nn.Linear(input_size, output_size)
            for param in self.model.cls.entangle_params_transform_layer.parameters():
                param.requires_grad = False 
        
        self.unitary_params = torch.nn.Parameter(torch.Tensor(self.num_unitary_circuit_layers, self.num_qubits, 3))

        self.model.cls.unitary_params.requires_grad = False
        for param in self.model.cls.predictions.parameters():
            param.requires_grad = False 
        for param in self.model.cls.out.parameters():
            param.requires_grad = False 
        for name, param in self.model.named_parameters():
            print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
        

        # Freeze classical-to-quantum projection net when asked
        if not projection_trainable:
            for param in self.entangle_params_transform_layer.parameters():
                param.requires_grad = False

        if not self.hparams.task_name == 'mc' and not self.hparams.task_name == 'rp':
            self.metric = datasets.load_metric(
                "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            )
        self.num_qubits = self.model.cls.num_qubits
        self.output_layer = nn.Linear(2**self.num_qubits, self.num_labels)
        self.outputs = defaultdict(list)
        self.quantum_layer = self.get_circuit()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        
        if self.pretrained:
            sequence_output= self.model.bert(input_ids=input_ids, \
                                              token_type_ids=token_type_ids, \
                                              attention_mask=attention_mask)[:1][0]
        elif self.pretrained == False:
        
            sequence_output, pooled_output = self.bert(input_ids=input_ids, \
                                        token_type_ids=token_type_ids, \
                                        attention_mask=attention_mask)
        
        entangle_params = self.entangle_params_transform_layer(sequence_output[:, 0])
        entangle_params = torch.sigmoid(entangle_params.reshape(-1,self.model.cls.num_entangle_circuit_layers,self.model.cls.num_qubits))* 2 * math.pi
    
        qcircuit_output = self.quantum_layer(entangle_params, self.unitary_params)
        output_logits = self.output_layer(qcircuit_output.float())
        if self.num_labels == 1:
            problem_type = "regression"
        elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            problem_type = "single_label_classification"
        else:
            problem_type = "multi_label_classification"

        if problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(output_logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(output_logits, labels)
        elif problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(output_logits.view(-1, self.num_labels), labels.view(-1))
        elif problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(output_logits, labels)

        output = (output_logits,)
        
        return ((loss,) + output) if loss is not None else output

    def training_step(self, batch, batch_idx):
        self.bitflip_prob = 0
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.bitflip_prob = self.config.bitflip_prob
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        self.outputs[dataloader_idx].append({"loss": val_loss, "preds": preds, "labels": labels})

    def on_validation_epoch_end(self):
        if self.hparams.task_name == "mnli":
            for i, outputs in self.outputs.items():
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in outputs]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            

        flat_outputs = []
        for lst in self.outputs.values():
            flat_outputs.extend(lst)

        preds = torch.cat([x["preds"] for x in flat_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in flat_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in flat_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)

        # if task is mc or rp
        if self.hparams.task_name == "mc" or self.hparams.task_name == "rp":
            pred_label = []
            pred_label.extend([0 if i < 0.5 else 1 for i in preds])
            pred_label = np.array(pred_label)
            correct_predictions = np.sum(pred_label == labels)
            total_predictions = len(pred_label)
            accuracy = correct_predictions / total_predictions
            reported_metric = {'accuracy': accuracy}
            self.log_dict(reported_metric , prog_bar=True)
        else:
            self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        self.outputs.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer_grouped_parameters = [
            {
                "params":[p for n, p in self.entangle_params_transform_layer.named_parameters()] +\
                    [p for n, p in self.output_layer.named_parameters()] +\
                    [self.unitary_params],  
                "weight_decay": self.hparams.weight_decay,
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
       

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
