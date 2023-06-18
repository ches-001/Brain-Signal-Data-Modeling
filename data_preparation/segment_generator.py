import os, glob, h5py, asyncio, tqdm, zipfile, argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from typing import Iterable, Callable, Any


class Queue:
    def __init__(self):
        self.paths = list()
        self.tasks = list()
        self.datatypes = list()
        self.class_names = list()

    def __len__(self):
        if len(self.paths) == len(self.tasks) == len(self.class_names):
            return len(self.paths)
        
        raise Exception(
            (
                f"paths, tasks and classes attributes are not of same length, "
                f"got {len(self.paths)}, {len(self.tasks)} and {len(self.class_names)} "
                "for each."
            )
        )
        
    def enqueue(self, path: str, task: str, datatype: str, class_name: str):
        self.paths.append(path)
        self.tasks.append(task)
        self.datatypes.append(datatype)
        self.class_names.append(class_name)
    
    def dequeue(self, idx:str):
        self.paths.pop(idx)
        self.tasks.pop(idx)
        self.datatypes.pop(idx)
        self.class_names.pop(idx)
    
    def __repr__(self):
        return f'{self.paths}'
    
    def __getiitem__(self, idx: int):
        return self.paths[idx], self.tasks[idx], self.datatypes, self.class_names[idx]
    
    def save(self, path: str) -> pd.DataFrame:
        df = pd.DataFrame()
        df["path"] = self.paths
        df["task"] = self.tasks
        df["datatype"] = self.datatypes
        df["class_name"] = self.class_names
        df.to_csv(path, index=False)
        return df
    

def extract_files(directory: str, delete_after: bool=True):
    for file_name in tqdm.tqdm(os.listdir(directory)):
        if file_name.endswith('.zip'):
            file_path = os.path.join(directory, file_name)

            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                extract_to = file_path.replace(".zip", "")
                if not os.path.isdir(extract_to):
                    os.mkdir(extract_to)

                zip_ref.extractall(extract_to)

            if delete_after:
                os.remove(file_path)


def segment_eeg_sample(sample_folder: str, queue: Queue):
    mat_files = glob.glob(os.path.join(sample_folder, "*.mat"), recursive=False)
    nback_files = [file for file in mat_files if "nback.mat" in file]
    dsr_files = [file for file in mat_files if "dsr.mat" in file]
    wg_files = [file for file in mat_files if "wg.mat" in file]

    tasks_files = [nback_files, dsr_files, wg_files]
    task_labels = ["nback", "dsr", "wg"]
    for tf, tl in zip(tasks_files, task_labels):
        cnt = [_ for _ in tf if "cnt_" in _][0]
        mrk = [_ for _ in tf if "mrk_" in _][0]

        sample_cnt = loadmat(cnt)
        sample_mrk = loadmat(mrk)
        cnt_key = list(sample_cnt.keys())[-1]
        sample_cnt = sample_cnt[cnt_key]
        mrk_key = list(sample_mrk.keys())[-1]
        sample_mrk = sample_mrk[mrk_key]

        # create sub folder to store segments / epochs
        task_segments_path = os.path.join(sample_folder, f"{tl}_segments")
        if not os.path.isdir(task_segments_path): 
            os.mkdir(task_segments_path)

        title = sample_cnt["title"][0][0][0]                                       # title
        cnt_sfreq = sample_cnt["fs"][0][0][0][0]                                   # sample frequency
        cnt_signals = sample_cnt["x"][0][0]
        cnt_event_onset = sample_mrk["time"][0][0].reshape(-1)                     # event / epoch onsets (millisecs)
        cnt_event_classes = sample_mrk["event"][0][0][0][0][0].reshape(-1)         # classes
        cnt_ohe_event_classes= sample_mrk["y"][0][0].T                             # one hot encoded classes
        class_names = sample_mrk["className"][0][0].reshape(-1)                    # class names
        class_names = [i.tolist()[0] for i in class_names.tolist()]
        time_indices = ((cnt_event_onset / 1000) * cnt_sfreq).astype(int).tolist() # time indices of each event onset
        time_indices.append(cnt_signals.shape[0])                                  # to capture the last segment


        start_times = time_indices[:-1]
        end_times = time_indices[1:]

        zipped = zip(start_times, end_times, cnt_event_classes, cnt_ohe_event_classes)
        for i, (start, end, event_label, ohe_event_label) in enumerate(zipped):

            segment_file = os.path.join(task_segments_path, f"seg{i}.h5")

            if not os.path.isfile(segment_file):
                #save each segment in a h5 file
                with h5py.File(segment_file, "w") as hdf_file:
                    group = hdf_file.create_group("data")
                    group["timestep_indices"] = np.array([start, end])
                    group["x"] = cnt_signals[start:end]
                    group["y"] = event_label
                    group["ohe_y"] = ohe_event_label
                    group["class_names"] = class_names
                hdf_file.close()

            class_idx = np.nonzero(ohe_event_label)[0][0]
            queue.enqueue(segment_file, title, "EEG", class_names[class_idx])


def segment_nirs_sample(sample_folder: str, queue: Queue):
    mat_files = glob.glob(os.path.join(sample_folder, "*.mat"), recursive=False)
    nback_files = [file for file in mat_files if "nback.mat" in file]
    dsr_files = [file for file in mat_files if "dsr.mat" in file]
    wg_files = [file for file in mat_files if "wg.mat" in file]

    tasks_files = [nback_files, dsr_files, wg_files]
    task_labels = ["nback", "dsr", "wg"]
    for tf, tl in zip(tasks_files, task_labels):
        cnt = [_ for _ in tf if "cnt_" in _][0]
        mrk = [_ for _ in tf if "mrk_" in _][0]

        sample_cnt = loadmat(cnt)
        sample_mrk = loadmat(mrk)
        cnt_key = list(sample_cnt.keys())[-1]
        sample_cnt = sample_cnt[cnt_key]
        mrk_key = list(sample_mrk.keys())[-1]
        sample_mrk = sample_mrk[mrk_key]

        # get oxy and deoxy data
        sample_oxy = sample_cnt["oxy"][0][0]
        sample_deoxy = sample_cnt["deoxy"][0][0]

        # create sub folder to store segments / epochs
        task_segments_path = os.path.join(sample_folder, f"{tl}_segments")
        if not os.path.isdir(task_segments_path): 
            os.mkdir(task_segments_path)

        title = sample_oxy["title"][0][0][0]                                         # title
        oxy_signals = sample_oxy["x"][0][0]                                          # Oxygenated hemoglobin (oxy-Hb):
        deoxy_signals = sample_deoxy["x"][0][0]                                      # Deoxygenated hemoglobin (deoxy-Hb):
        cnt_sfreq = sample_oxy["fs"][0][0][0][0]                                     # sample frequency
        cnt_event_onset = sample_mrk["time"][0][0].reshape(-1)                       # event / epoch onsets (millisecs)
        cnt_event_classes = sample_mrk["event"][0][0][0][0][0].reshape(-1)           # classes
        cnt_ohe_event_classes= sample_mrk["y"][0][0].T                               # one hot encoded classes
        class_names = sample_mrk["className"][0][0].reshape(-1)                      # class names
        class_names = [i.tolist()[0] for i in class_names.tolist()]
        time_indices = ((cnt_event_onset / 1000) * cnt_sfreq).astype(int).tolist()   # time indices of each event onset
        time_indices.append(oxy_signals.shape[0])                                    # to capture the last segment


        start_times = time_indices[:-1]
        end_times = time_indices[1:]

        zipped = zip(start_times, end_times, cnt_event_classes, cnt_ohe_event_classes)
        for i, (start, end, event_label, ohe_event_label) in enumerate(zipped):

            segment_file = os.path.join(task_segments_path, f"seg{i}.h5")

            if not os.path.isfile(segment_file):            
                #save each segment in a h5 file
                with h5py.File(segment_file, "w") as hdf_file:
                    group = hdf_file.create_group("data")
                    group["timestep_indices"] = np.array([start, end])
                    group["x_oxy"] = oxy_signals[start:end]
                    group["x_deoxy"] = deoxy_signals[start:end]
                    group["y"] = event_label
                    group["ohe_y"] = ohe_event_label
                    group["class_names"] = class_names
                hdf_file.close()

            class_idx = np.nonzero(ohe_event_label)[0][0]
            queue.enqueue(segment_file, title, "NIRS", class_names[class_idx])


async def segmentor_coroutine(
        base_func: str, 
        *,
        args: Iterable[Any],
        semaphore: asyncio.Semaphore):
    
    running_loop = asyncio.get_running_loop()    
    async with semaphore:
        await running_loop.run_in_executor(None, lambda : base_func(*args))


async def main(
        base_dir: str, 
        base_func: Callable,
        *,
        queue: Queue,
        n_concurrency: int=10):
    
    semaphore = asyncio.Semaphore(n_concurrency)
    tasks = [
        segmentor_coroutine(
            base_func, 
            args=(os.path.join(base_dir, sample_dir), queue), 
            semaphore=semaphore
        ) 
        for sample_dir in os.listdir(base_dir)
    ]
    [await _ for _ in tqdm.tqdm(asyncio.as_completed(tasks))]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="\n%(asctime)s - %(levelname)s - %(message)s")

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    eeg_data_dir = os.path.join(script_dir, "../data/samples/EEG")
    nirs_data_dir = os.path.join(script_dir, "../data/samples/NIRS")
    eeg_csv_path = os.path.join(script_dir, "../data/samples/EEG.csv")
    nirs_csv_path = os.path.join(script_dir, "../data/samples/NIRS.csv")
    n_concurrency = 20

    # CLI Parsed parameters
    parser = argparse.ArgumentParser(description=f"Segments Generator")
    parser.add_argument("--eeg_path", type=str, default=eeg_data_dir, metavar='', help='EEG data path')
    parser.add_argument("--nirs_path", type=str, default=nirs_data_dir, metavar='', help='NIRS data path')
    parser.add_argument("--eeg_csv_path", type=str, default=eeg_csv_path, metavar='', help='EEG csv meta data path')
    parser.add_argument("--nirs_csv_path", type=str, default=nirs_csv_path, metavar='', help='NIRS csv meta data path')
    parser.add_argument("--n_concurrency", type=int, default=n_concurrency, metavar='', help='Number of concurrent coroutines')
    args = parser.parse_args()

    eeg_data_dir = args.eeg_path
    nirs_data_dir = args.nirs_path
    n_concurrency = args.n_concurrency
    eeg_csv_path = args.eeg_csv_path
    nirs_csv_path = args.nirs_csv_path

    try:
        eeg_queue = Queue()
        logging.info("Extracting EEG samples:")
        extract_files(eeg_data_dir)
        logging.info("Generating EEG segments:")
        asyncio.run(main(eeg_data_dir, segment_eeg_sample, queue=eeg_queue, n_concurrency=n_concurrency))
        eeg_queue.save(eeg_csv_path)
    except Exception as e:
        logging.error(e)

    try:
        nirs_queue = Queue()
        logging.info("Extracting NIRS samples:")
        extract_files(nirs_data_dir)
        logging.info("Generating NIRS segments:")
        asyncio.run(main(nirs_data_dir, segment_nirs_sample, queue=nirs_queue, n_concurrency=n_concurrency))
        nirs_queue.save(nirs_csv_path)
    except Exception as e:
        logging.error(e)