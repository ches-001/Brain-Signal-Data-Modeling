import torch, h5py, os, glob, h5py, random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from typing import Optional, Tuple, Union, Iterable

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
            excluded: Optional[Iterable[str]] = None,
            sample_size: Optional[Union[int, float]] = None,
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
        self.excluded = excluded
        self.segment_files = self._get_segment_paths(excluded, sample_size)
    
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
        input_signal = self._resize(input_signal).unsqueeze(dim=0).float()  # shape: (1, n_channels, time)
        label = torch.from_numpy(label)                                     # one hot encoded
        label = torch.nonzero(label, as_tuple=False).squeeze().long()       # label encoded

        return input_signal, label
    
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