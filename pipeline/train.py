import os, torch, tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from matplotlib import pyplot as plt
from typing import Any, Dict, Iterable, Tuple

class TrainingPipeline:
    def __init__(self, 
                model: nn.Module,
                lossfunc: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: str="cpu", 
                weight_init: bool=True,
                custom_weight_initializer: Any=None,
                dirname: str="./saved_model", 
                filename: str="model.pth.tar",
                save_metrics: bool=True, 
                onehot_labels: bool=False):
        
        self.device = device
        self.model = model.to(self.device)
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.custom_weight_initializer = custom_weight_initializer
        self.dirname = dirname
        self.filename = filename
        self.save_metrics = save_metrics
        self.onehot_labels = onehot_labels
        
        if self.weight_init:
            if self.custom_weight_initializer:
                self.model.apply(self.custom_weight_initializer)
            else:
                self.model.apply(self.xavier_init_weights)

        # collect metrics in this dictionary
        if self.save_metrics:
            self._train_metrics_dict = dict(loss=[], accuracy=[], f1=[], precision=[], recall=[])
            self._eval_metrics_dict = dict(loss=[], accuracy=[], f1=[], precision=[], recall=[])
        
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
    
    def collect_metric(self) -> Tuple[Dict[str, Iterable[float]], Dict[str, Iterable[float]]]:
        if self.save_metrics:
            return self._train_metrics_dict, self._eval_metrics_dict
        
    def plot_metrics(
            self, 
            mode: str,
            figsize: Tuple[float, float]=(20, 6)):
        
        valid_modes = self._valid_modes()
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        
        _, axs = plt.subplots(1, 2, figsize=figsize)
        axs[0].plot(getattr(self, f"_{mode}_metrics_dict")["loss"])
        axs[0].set_title(f"{mode} Loss")

        for k in self._train_metrics_dict.keys():
            if k == "loss": continue
            axs[1].plot(getattr(self, f"_{mode}_metrics_dict")[k], label=f"{k.title()}")
            axs[1].legend()
        axs[1].set_title(f"{mode} Performance")
        plt.show()
        print("\n\n")
        
    def train(self, dataloader: DataLoader, verbose: bool=False):
        return self._feed(dataloader, "train", verbose)
    
    def evaluate(self, dataloader: DataLoader, verbose: bool=False):        
        with torch.no_grad():
            return self._feed(dataloader, "eval", verbose)
        
    def _feed(self, dataloader: DataLoader, mode: str, verbose: bool=False):
        assert mode in self._valid_modes(), "Invalid Mode"
        getattr(self.model, mode)()
        loss, acc, f1, precision, recall = 0, 0, 0, 0, 0
        
        for idx, (batch) in tqdm.tqdm(enumerate(dataloader)):
            if dataloader.dataset.signal != "multimodal":
                signals, labels = batch
                signals = signals.to(self.device)
                labels = labels.to(self.device)           
                probs = self.model(signals)

            else:
                eeg_signals, nirs_oxy_signals, nirs_deoxy_signals, labels = batch
                eeg_signals = eeg_signals.to(self.device)
                nirs_oxy_signals = nirs_oxy_signals.to(self.device)
                nirs_deoxy_signals = nirs_deoxy_signals.to(self.device)
                labels = labels.to(self.device)         
                probs = self.model(eeg_signals, nirs_oxy_signals, nirs_deoxy_signals)

            batch_loss = self.lossfunc(probs, labels)
            
            if mode == "train":
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            preds = probs.argmax(dim=1).cpu().numpy()
            if self.onehot_labels:
                labels = labels.argmax(dim=1).cpu().numpy()
            else:
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
            
        if self.save_metrics:
            getattr(self, f"_{mode}_metrics_dict")["loss"].append(loss)
            getattr(self, f"_{mode}_metrics_dict")["accuracy"].append(acc)
            getattr(self, f"_{mode}_metrics_dict")["f1"].append(f1)
            getattr(self, f"_{mode}_metrics_dict")["precision"].append(precision)
            getattr(self, f"_{mode}_metrics_dict")["recall"].append(recall)

        return loss, acc, f1, precision, recall
    
    def _valid_modes(self) -> Iterable[str]:
        return ["train", "eval"]