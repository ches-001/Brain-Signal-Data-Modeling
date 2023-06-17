import torch, h5py, os, glob, h5py, random
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn import functional as F
from typing import Optional, Tuple, Union, Iterable
from torchaudio.transforms import Spectrogram, AmplitudeToDB

class SignalDataset(Dataset):
    def __init__(
            self,
            base_dir: str,
            signal: str,
            task: str,
            hemoglobin: Optional[str]=None,
            scale: bool = True,
            scale_range: Tuple[float, float] = (0.0, 1.0),
            t_size: int = 200,
            shuffle: bool = True,
            excluded: Optional[Iterable[str]] = None,
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
        self.hemoglobin = hemoglobin
        self.scale = scale
        self.scale_range = scale_range
        self.t_size = t_size
        self.shuffle = shuffle
        self.excluded = excluded
        self.onehot_labels = onehot_labels
        self.use_spectrogram = use_spectrogram
        self.avg_spectrogram_ch = avg_spectrogram_ch
        self.segment_files = self._get_segment_paths(excluded, sample_size)

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
        return len(self.segment_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        segment = self.segment_files[idx]
        input_signal = None
        label = None

        with h5py.File(segment, "r") as hdf_file:
            if self.signal == "EEG":
                input_signal = hdf_file["data"]["x"]
            else:
                input_signal = hdf_file["data"][f"x_{self.hemoglobin}"]

            input_signal = np.array(input_signal)
            label = np.array(hdf_file["data"]["ohe_y"])
        hdf_file.close()

        if self.scale:
            input_signal = self._scale_input(input_signal)
        input_signal = torch.from_numpy(input_signal)
        input_signal = input_signal.permute(1, 0)                           # shape: (time, n_channels) -> (n_channels, time)
        input_signal = self._resize(input_signal).unsqueeze(dim=0).float()  # shape: (1, n_channels, t_size)
        
        if self.use_spectrogram:
            input_signal = self.spectrogram_model(input_signal).squeeze()   #shape: (n_channels, sh, sw)
            if self.avg_spectrogram_ch:
                input_signal = input_signal.mean(dim=0).unsqueeze(dim=0)    #shape: (1, sh, sw)
                
        label = torch.from_numpy(label).float()                             # onehot encoded
        if not self.onehot_labels:
            label = (
                torch
                .nonzero(label, as_tuple=False)
                .squeeze()
                .long()
            )                                                               # label encoded
        return input_signal, label
    
    def get_sample_classes(self) -> pd.DataFrame:
        if self._class_df_cache is not None:
            return self._class_df_cache
        
        class_df = pd.DataFrame()
        if self.onehot_labels:
            get_class_val = lambda i : (
                torch
                .nonzero((self.__getitem__(i)[1]), as_tuple=False)
                .squeeze()
                .long()
                .item()
            )
        else:
            get_class_val = lambda i : self.__getitem__(i)[1].item()
        class_df["class_label"] = [get_class_val(i) for i in range(self.__len__())]
        self._class_df_cache = class_df

        return class_df
    
    def get_sample_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        classes_df = self.get_sample_classes()
        unique_classes = classes_df["class_label"].unique()
        
        class_weights = torch.zeros(unique_classes.shape)
        for i, c in enumerate(unique_classes):
            # smaller classes will have more weights
            class_weights[i] = len(classes_df) / len(classes_df[classes_df["class_label"]==c])
        class_weights /= class_weights.max()

        sample_weights = torch.zeros(classes_df.shape[0])
        for i in range(classes_df.shape[0]):
            label = classes_df["class_label"].iloc[i]
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
        with h5py.File(self.segment_files[0], "r") as hdf_file:
            class_names = list(hdf_file["data"]["class_names"])
        hdf_file.close()
        return class_names

    def _get_segment_paths(
            self, 
            excluded: Optional[Iterable[str]],
            sample_size: Optional[Union[int, float]]) -> Iterable[str]:
        
        segments_foldername = f"{self.task}_segments"
        segments = []
        for sample_dir in os.listdir(self.data_dir):
            sample_dir = os.path.join(self.data_dir, sample_dir)
            segment_dir  = os.path.join(sample_dir, segments_foldername)
            segment_files = glob.glob(os.path.join(segment_dir, "*.h5"), recursive=False)
            segments.extend(segment_files)

        if excluded is not None:
            # exclude some segments if any
            segments = [segment for segment in segments if segment not in excluded]

        if self.shuffle:
            random.shuffle(segments)
            
        if sample_size is not None:
            if isinstance(sample_size, int):
                return segments[:sample_size]
            elif isinstance(sample_size, float):
                sample_size = int(sample_size * len(segments))
                return segments[:sample_size]
            else:
                raise ValueError("Invalid value for sample_size")
        return segments