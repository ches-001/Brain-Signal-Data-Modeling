import os, torch, tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from typing import Any

class TrainingPipeline:
    def __init__(self, 
                model: nn.Module,
                lossfunc: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: str="cpu", 
                weight_init: bool=True,
                custom_weight_initializer: Any=None,
                dirname: str="./saved_model", 
                filename: str="model.pth.tar"):
        
        self.device = device
        self.model = model.to(self.device)
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.custom_weight_initializer = custom_weight_initializer
        self.dirname = dirname
        self.filename = filename
        
        if self.weight_init:
            if self.custom_weight_initializer:
                self.model.apply(self.custom_weight_initializer)
            else:
                self.model.apply(self.xavier_init_weights)
        
    def xavier_init_weights(self, m: nn.Module):
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)) and (m.weight.requires_grad == True):
            nn.init.xavier_uniform_(m.weight)
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)
    
    def save_model(self):
        if not os.path.isdir(self.dirname): os.mkdir(self.dirname)
        state_dicts = {
            "network_params":self.model.state_dict(),
            "optimizer_params":self.optimizer.state_dict(),
        }
        return torch.save(state_dicts, os.path.join(self.dirname, self.filename))
        
    def train(self, dataloader: DataLoader, verbose: bool=False):
        loss, acc, f1, precision, recall = self._feed(dataloader, "train", verbose)
        return loss, acc, f1, precision, recall
    
    def evaluate(self, dataloader: DataLoader, verbose: bool=False):        
        with torch.no_grad():
            loss, acc, f1, precision, recall = self._feed(dataloader, "eval", verbose)
            return loss, acc, f1, precision, recall
        
    def _feed(self, dataloader: DataLoader, mode: str, verbose: bool=False):
        assert mode in ["train", "eval"], "Invalid Mode"
        getattr(self.model, mode)()
        loss, acc, f1, precision, recall = 0, 0, 0, 0, 0
        
        for idx, (signals, labels) in tqdm.tqdm(enumerate(dataloader)):
            signals = signals.to(self.device)       #shape: (N, n_channels, n_time)
            labels = labels.to(self.device)         #shape: (N, 1)    
                        
            probs = self.model(signals)
            batch_loss = self.lossfunc(probs, labels)
            
            if mode == "train":
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            preds = probs.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            batch_acc = accuracy_score(preds, labels)
            
            loss += batch_loss.item()
            acc += batch_acc
            f1 += f1_score(labels, preds, average="macro")
            precision += precision_score(labels, preds, average="macro")    
            recall += recall_score(labels, preds, average="macro")

        loss /= (idx + 1)
        acc /= (idx + 1)
        f1 /= (idx + 1)
        precision /= (idx + 1)
        recall /= (idx + 1)

        verbosity_label = mode.title()
        if verbose:
            print((
                f"{verbosity_label} Loss: {round(loss, 4)} \t{verbosity_label} Accuracy: {round(acc, 4)}"
                f"\t{verbosity_label} F1: {round(f1, 4)} \t{verbosity_label} Precision: {round(precision, 4)}"
                f"\t{verbosity_label} Recall: {round(recall, 4)}"
                ))

        return loss, acc, f1, precision, recall