import torch, h5py, os, h5py
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn import functional as F
from typing import Optional, Tuple, Union, Iterable, Callable, Any
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class SignalDataset(Dataset):
    def __init__(
            self,
            base_dir: str,
            *,
            signal: str,
            task: str,
            meta_df: pd.DataFrame,
            hemoglobin: Optional[str]=None,
            scale: bool = True,
            scale_range: Tuple[float, float] = (0.0, 1.0),
            t_size: int = 200,
            t_eeg_size: Optional[int] = None,
            t_nirs_size: Optional[int] = None,
            transforms: Optional[Callable]=None,
            transforms_p: float=0.0,
            excluded_paths: Optional[Iterable[str]] = None,
            excluded_classes: Optional[Iterable[str]] = None,
            sample_size: Optional[Union[int, float]] = None,
            use_rp: bool=False,
            rp_threshold: float=0.2,
            average_rp_channels: bool=True,
            onehot_labels: bool=False,
        ):

        valid_signals = ("EEG", "NIRS", "multimodal")
        if not signal or signal not in valid_signals:
            raise ValueError(f"signal can only be one of {valid_signals}, got {signal}")
        
        valid_tasks = ("nback", "dsr", "wg")
        if not task or task not in valid_tasks:
            raise ValueError(f"tasl can only be one of {(valid_tasks)}, got {task}")
        
        if signal == "NIRS":
            valid_hemoglobin_type = ("oxy", "deoxy")
            if hemoglobin is None or hemoglobin not in valid_hemoglobin_type:
                raise ValueError(f"hemoglobin should be one of {valid_hemoglobin_type}, got {hemoglobin}")
        
        if not os.path.isdir(base_dir):
            raise OSError(f"No such directory {base_dir}")

        self.base_dir = base_dir
        self.signal = signal
        self.task = task
        self.meta_df = meta_df
        self.hemoglobin = hemoglobin
        self.scale = scale
        self.scale_range = scale_range
        self.t_size = t_size
        self.t_eeg_size = t_eeg_size
        self.t_nirs_size = t_nirs_size
        self.transforms = transforms
        self.transforms_p = transforms_p
        self.onehot_labels = onehot_labels
        self.use_rp = use_rp
        self.rp_threshold = rp_threshold
        self.average_rp_channels = average_rp_channels
        self.meta_df = self._process_df(meta_df, excluded_paths, excluded_classes, sample_size)

        self.class_label_encoder = LabelEncoder()
        self.class_labels = self.class_label_encoder.fit_transform(self.meta_df["class_name"])

        self.class_oh_encoder = OneHotEncoder()
        self.class_ohe = self.class_oh_encoder.fit_transform(self.class_labels.reshape(-1, 1))

        self._class_df_cache = None
    

    def __len__(self) -> int:
        return len(self.meta_df)
    

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.signal == "EEG" or self.signal == "NIRS":
            segment_path = self.meta_df["path"].iloc[idx]
            return self._get_sample_epoch(idx, segment_path, self.signal, hemoglobin=self.hemoglobin, t_size=self.t_size)

        elif self.signal == "multimodal":
            eeg_segment_path = self.meta_df["eeg_path"].iloc[idx]
            nirs_segment_path = self.meta_df["nirs_path"].iloc[idx]
            eeg_signal, _ = self._get_sample_epoch(idx, eeg_segment_path, "EEG", t_size=self.t_eeg_size)
            nirs_oxy_signal, label = self._get_sample_epoch(idx, nirs_segment_path, "NIRS", hemoglobin="oxy", t_size=self.t_nirs_size)
            nirs_deoxy_signal, label = self._get_sample_epoch(idx, nirs_segment_path, "NIRS", hemoglobin="deoxy", t_size=self.t_nirs_size)
            return eeg_signal, nirs_oxy_signal, nirs_deoxy_signal, label
        else:
            pass

    def _get_sample_epoch(
            self, idx: int,
            segment_path: str, 
            signal: str, 
            hemoglobin: Optional[str]=None, 
            t_size: Optional[int]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        with h5py.File(segment_path, "r") as hdf_file:
            if signal == "EEG":
                input_signal = hdf_file["data"]["x"]
            else:
                input_signal = hdf_file["data"][f"x_{hemoglobin}"]

            input_signal = np.array(input_signal)
        hdf_file.close()

        if self.onehot_labels:
            label = self.class_ohe[idx].toarray()
            label = torch.from_numpy(label).squeeze().float()
        else:
            label = self.class_labels[idx]
            label = torch.tensor(label).long()

        if self.scale:
            input_signal = self._scale_input(input_signal)
        input_signal = torch.from_numpy(input_signal)
        input_signal = input_signal.permute(1, 0)                         # shape: (time, n_channels) -> (n_channels, time)
        input_signal = self._resize(input_signal, t_size).float()         # shape: (n_channels, t_size)
        
        if self.transforms:
            if self.transforms_p > np.random.random():
                input_signal = self.transforms(input_signal.unsqueeze(dim=0)).squeeze()

        if self.use_rp:
            # generate recurrence plot and average channels
            input_signal = torch.abs(input_signal.unsqueeze(1) - input_signal.unsqueeze(2))
            input_signal[input_signal > self.rp_threshold] = 1
            input_signal[input_signal <= self.rp_threshold] = 0
            if self.average_rp_channels:
                input_signal = input_signal.mean(dim=0).unsqueeze(dim=0)  # shape: (1, t_size, t_size)

        return input_signal, label
    

    def get_sample_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_classes = self.class_labels
        unique_classes = np.unique(sample_classes)
        
        n_classes = len(unique_classes)
        class_weights = torch.zeros(n_classes)
        for i, c in enumerate(unique_classes):
            # smaller classes will have more weights
            class_weights[i] = len(sample_classes) / len(sample_classes[sample_classes==c])
        class_weights /= class_weights.max()

        n_samples = len(sample_classes)
        sample_weights = torch.zeros(n_samples)
        for i in range(n_samples):
            label = sample_classes[i]
            sample_weights[i] = class_weights[label]
        
        return class_weights, sample_weights
    

    def _scale_input(self, input: np.ndarray) -> np.ndarray:
        # input shape: (time, n_channels)
        output = (input - input.min()) / (input.max() - input.min())
        a, b = self.scale_range
        output = (b - a) * output + a
        return output
    

    def _resize(self, input: torch.Tensor, t_size: Optional[int]=None) -> torch.Tensor:
        if not t_size: 
            return input
        # input shape: (n_channels, time)
        c, t = input.shape
        input = input.unsqueeze(dim=0)
        output =  F.interpolate(input, size=(t_size), mode="linear", align_corners=False)
        # input shape: (n_channels, t_size)
        return output.squeeze()
    

    def get_label_names(self) -> Iterable[str]:
        class_names = self.meta_df["class_name"].unique()
        return class_names
    

    def _process_df(
            self, 
            meta_df: pd.DataFrame,
            excluded_paths: Optional[Iterable[str]]=None, 
            excluded_classes: Optional[Iterable[str]]=None,
            sample_size: Optional[Union[int, float]]=None) -> pd.DataFrame:
        
        if self.signal == "multimodal":
            if "eeg_path" not in meta_df.columns or "nirs_path" not in meta_df.columns:
                raise Exception(
                    "For if signal is multimodal, meta_df must have 'eeg_path' and 'nirs_path' columns"
                )
        df = meta_df.copy()
        if excluded_paths is not None:
            if self.signal=="EEG" or self.signal=="NIRS":
                df = df[~df["path"].isin(excluded_paths)]

            elif self.signal == "multimodal":
                if "EEG" in excluded_paths[0]:
                    df = df[~df["eeg_path"].isin(excluded_paths)]
                elif "NIRS" in excluded_paths[0]:
                    df = df[~df["nirs_path"].isin(excluded_paths)]
                else:
                    pass
            else:
                pass

        if self.signal != "multimodal":
            df = df[df["datatype"] == self.signal]
        df = df[df["task"] == self.task]
        
        if excluded_classes is not None:
            df = df[~df["class_name"].isin(excluded_classes)]
        if sample_size is not None:
            if isinstance(sample_size, int):
                return df.iloc[:sample_size]
            elif isinstance(sample_size, float):
                sample_size = int(sample_size * len(df))
                return df.iloc[:sample_size]
            else:
                raise ValueError("Invalid value for sample_size")
        return df