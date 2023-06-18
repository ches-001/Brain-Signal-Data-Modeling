import torch, h5py, os, glob, h5py, random
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn import functional as F
from typing import Optional, Tuple, Union, Iterable, Callable, Any
from torchaudio.transforms import Spectrogram, AmplitudeToDB
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
            transforms: Optional[Callable]=None,
            transforms_p: float=0.0,
            excluded_paths: Optional[Iterable[str]] = None,
            excluded_classes: Optional[Iterable[str]] = None,
            sample_size: Optional[Union[int, float]] = None,
            use_spectrogram: bool=False,
            onehot_labels: bool=False,
            avg_spectrogram_ch: bool = True,
            **spectrogram_kwargs,
        ):

        valid_signals = ("EEG", "NIRS")
        if not signal or signal not in valid_signals:
            raise ValueError(f"signal can only be one of {valid_signals}, got {signal}")
        
        valid_tasks = ("nback", "dsr", "wg")
        if not task or task not in valid_tasks:
            raise ValueError(f"tasl can only be one of {(valid_tasks)}, got {task}")
        
        if signal == "NIRS":
            valid_hemoglobin_type = ("oxy", "deoxy")
            if hemoglobin is None or hemoglobin not in valid_hemoglobin_type:
                raise ValueError(f"hemoglobin should be one of {valid_hemoglobin_type}, got {hemoglobin}")
            
        data_dir = os.path.join(base_dir, signal)
        if not os.path.isdir(data_dir):
            raise OSError(f"No such directory {data_dir}")

        self.base_dir = base_dir
        self.data_dir = data_dir
        self.signal = signal
        self.task = task
        self.meta_df = meta_df
        self.hemoglobin = hemoglobin
        self.scale = scale
        self.scale_range = scale_range
        self.t_size = t_size
        self.transforms = transforms
        self.transforms_p = transforms_p
        self.onehot_labels = onehot_labels
        self.use_spectrogram = use_spectrogram
        self.avg_spectrogram_ch = avg_spectrogram_ch
        self.meta_df = self._process_df(meta_df, excluded_paths, excluded_classes, sample_size)

        self.class_label_encoder = LabelEncoder()
        self.class_labels = self.class_label_encoder.fit_transform(self.meta_df["class_name"])

        self.class_oh_encoder = OneHotEncoder()
        self.class_ohe = self.class_oh_encoder.fit_transform(self.class_labels.reshape(-1, 1))

        # spectrogram arguments
        n_fft = int(self.t_size * 0.8)
        win_length = max(1, int(self.t_size * 0.1))
        hop_length = max(1, win_length // 3)
        self.spectrogram_kwargs = dict(
            n_fft=n_fft, 
            win_length=win_length, 
            hop_length=hop_length, 
            power=2, 
            normalized=True, 
            pad=0,
        )
        self.kwargs = {**spectrogram_kwargs, **self.spectrogram_kwargs}
        self.spectrogram_model = None
        if self.use_spectrogram:
            self.spectrogram_model = nn.Sequential(
                Spectrogram(**self.kwargs),
                AmplitudeToDB(stype="power" if (self.kwargs["power"]==2) else "magnitude"),
            )

        self._class_df_cache = None
    

    def __len__(self) -> int:
        return len(self.meta_df)
    

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        segment_path = self.meta_df["path"].iloc[idx]
        input_signal = None
        label = None

        with h5py.File(segment_path, "r") as hdf_file:
            if self.signal == "EEG":
                input_signal = hdf_file["data"]["x"]
            else:
                input_signal = hdf_file["data"][f"x_{self.hemoglobin}"]

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
        input_signal = input_signal.permute(1, 0)                           # shape: (time, n_channels) -> (n_channels, time)
        input_signal = self._resize(input_signal).unsqueeze(dim=0).float()  # shape: (1, n_channels, t_size)
        
        if self.use_spectrogram:
            input_signal = self.spectrogram_model(input_signal).squeeze()   #shape: (n_channels, sh, sw)
            if self.avg_spectrogram_ch:
                input_signal = input_signal.mean(dim=0).unsqueeze(dim=0)    #shape: (1, sh, sw)

        if self.transforms:
            if self.transforms_p > np.random.random():
                input_signal = self.transforms(input_signal)

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
    

    def _resize(self, input: torch.Tensor) -> torch.Tensor:
        # input shape: (n_channels, time)
        c, t = input.shape
        input = input.unsqueeze(dim=0)
        output =  F.interpolate(input, size=(self.t_size), mode="linear", align_corners=False)
        # input shape: (n_channels, self.t_size)
        return output.squeeze()
    

    def get_label_names(self) -> Iterable[str]:
        class_names = self.meta_df["class_name"].unique()
        return class_names
    

    def _process_df(
            self, meta_df: pd.DataFrame,
            excluded_paths: Optional[Iterable[str]], 
            excluded_classes: Optional[Iterable[str]],
            sample_size: Optional[Union[int, float]]) -> pd.DataFrame:
        
        df = meta_df.copy()
        if excluded_paths is not None:
            df = df[~df["path"].isin(excluded_paths)]

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