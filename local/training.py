import multiprocessing
import numpy as np
import h5py
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional, Callable
from itertools import pairwise
from operator import itemgetter
from pathlib import Path
from scipy.io.wavfile import write as wavwrite
from LPCNet import LPCNet


logger = logging.getLogger("training.py")


Transform = Optional[Callable]


# region Dataset class
class SequentialSpeechTrials(Dataset):
    """
    Dataset class that returns each trial individually, independent of the number of time steps (such as 15 seconds for
    sentences, or 2 seconds for single words).
    """
    def __init__(self, feature_files: List[str], transform: Transform = None, target_transform: Transform = None,
                 target_specifier: str = "lpc_coefficients"):
        self.feature_files = feature_files
        self.transform = transform
        self.target_transform = target_transform
        self.target_specifier = target_specifier

        # Open each feature file
        self.fhs = [h5py.File(hdf_file, 'r') for hdf_file in feature_files]
        self.nb_trials = [self._count_trials(fh['trial_ids'][...]) for fh in self.fhs]
        self.trial_labels = []
        self.trial_filename = []
        self.frame_counter = 0
        for fh, fname in zip(self.fhs, feature_files):
            self.frame_counter += len(fh['trial_ids'][...])
            trial_stimuli = self._squeeze_trial_ids(fh['trial_ids'][...])
            self.trial_labels.extend(trial_stimuli)
            self.trial_filename.extend([fname] * len(trial_stimuli))

        self.cumulative_length = 0
        self.feature_dict = {}  # Will hold data in form (start_index, stop_index): fh
        for nb_trials, fh in zip(self.nb_trials, self.fhs):
            self.feature_dict[(self.cumulative_length, self.cumulative_length + nb_trials)] = fh
            self.cumulative_length += nb_trials

    def __del__(self):
        for fh in self.fhs:
            fh.close()

    def __len__(self):
        return sum(self.nb_trials)

    @staticmethod
    def _count_trials(trial_ids: List[int]) -> int:
        return len(np.where(trial_ids[:-1] != trial_ids[1:])[0]) + 1

    @staticmethod
    def _find_indices_of_nth_subsequence(n: int, seq: np.ndarray) -> Tuple[int, int]:
        """
        Given a sequence of integer values, find the boundaries of the nth subsequence within that sequence. Example:
        seq: [4, 4, 4, 3, 3, 3, -3, -3, -3, 5, 5, 5, -5, -5, -5, 5, 5, 5], n=4 would result in (9, 12)

        Subsequences are zero-indexed.
        """
        take_nth_element = itemgetter(n)
        borders = (np.where(seq[:-1] != seq[1:])[0] + 1).tolist()
        borders = [0] + borders + [len(seq)]
        borders = tuple(pairwise(borders))
        start, stop = take_nth_element(borders)
        return start, stop

    @staticmethod
    def _squeeze_trial_ids(trial_ids: List[int]) -> List[int]:
        last_entry = trial_ids[0]
        result = [last_entry]
        for i in range(1, len(trial_ids)):
            if trial_ids[i] != last_entry:
                result.append(abs(trial_ids[i]))
                last_entry = trial_ids[i]

        return result

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        for (start, stop) in self.feature_dict.keys():
            if start <= index < stop:
                trial_ids = self.feature_dict[(start, stop)]['trial_ids'][...]
                trial_start, trial_stop = self._find_indices_of_nth_subsequence(index - start, trial_ids)

                hga = self.feature_dict[(start, stop)]['hga_activity'][trial_start:trial_stop]
                lpc = self.feature_dict[(start, stop)][self.target_specifier][trial_start:trial_stop]

                if self.transform:
                    hga = self.transform(hga)
                if self.target_transform:
                    lpc = self.target_transform(lpc)
                return hga, lpc

    def __repr__(self):
        covered_days = sorted(set([Path(feature_file).parent.name for feature_file in self.feature_files]))
        return f"SequentialSpeechTrials: {sum(self.nb_trials)} trials with {self.frame_counter} frames " \
               f"(total: {(self.frame_counter * 0.01) / 3600:.02f}h). Days covered: {', '.join(covered_days)}"

    def plot_trial(self, index: int, stimuli_map: Optional[Dict[int, str]] = None):
        hga, lpc = self[index]

        label = stimuli_map[self.trial_labels[index]] if stimuli_map is not None else str(self.trial_labels[index])
        ig, (ax_hga, ax_lpc) = plt.subplots(2, 1,figsize=(14, 8), num=1, clear=True)

        ax_hga.set_title(f"Label: {label}, Filename: {self.trial_filename[index]}", loc="left")
        hga_im = ax_hga.imshow(hga.T, aspect='auto', origin='lower', cmap='bwr', vmin=-4, vmax=4)
        ax_hga.set_xlim(0, len(hga))
        # ax_hga.set_xticks([])
        ax_hga.set_ylabel('Channel', labelpad=-18)
        ax_hga.set_yticks(np.linspace(0, hga.shape[1], 2, endpoint=True))
        ax_hga.set_yticklabels([1, hga.shape[1]])

        lpc_im = ax_lpc.imshow(lpc.T, aspect='auto', origin='lower', cmap='viridis')
        ax_lpc.set_xlim(0, len(lpc))
        # ax_lpc.set_xticks([])
        ax_lpc.set_ylabel('LPC coefficients', labelpad=-18)
        ax_lpc.set_yticks(np.linspace(0, lpc.shape[1] - 1, 2, endpoint=True))
        ax_lpc.set_yticklabels([1, 20])

        plt.show()
# endregion


class StoreBestModel:
    """
    Store the best model (according to the validation accuracy) at a dedicated location.
    The model will not be updated if the validation score is worse than any seen before.
    """
    def __init__(self, filename: str, info: Optional[dict] = None):
        self.current_best_validation_acc = -np.inf
        self.current_best_validation_loss = np.inf
        self.filename = filename
        self.optional_info = info

    def update(self, model: nn.Module, validation_acc: Optional[float] = None,
               validation_loss: Optional[float] = None, info: Optional[dict] = None):
        if validation_acc is not None and validation_loss is not None:
            logger.error(f"Class can only be used for either accuracy or loss.")
            exit(1)

        if validation_acc is not None and (validation_acc > self.current_best_validation_acc):
            torch.save(model.state_dict(), self.filename)
            self.current_best_validation_acc = validation_acc
            logger.info(f"Updated best model weights for a score of {validation_acc}.")
            self.optional_info = info

        if validation_loss is not None and (validation_loss < self.current_best_validation_loss):
            torch.save(model.state_dict(), self.filename)
            self.current_best_validation_loss = validation_loss
            logger.info(f"Updated best model weights for a score of {validation_loss}.")


class AsynchronousSynthesisQueue:
    """
    Class for synthesizing intermediate results on different CPU cores. Intermediate results have to be stored as .npy
    files. Each of the given .npy files will be converted to .wav files by LPCNet with the same filename (but different)
    file extension.
    """
    def __init__(self, nb_processes: int):
        self.pool = multiprocessing.Pool(processes=nb_processes)

    def wait(self):
        """
        Wait for all synthesis jobs to be finished before closing the pool.
        """
        self.pool.close()
        self.pool.join()

    @staticmethod
    def _generate_audio_from_lpc(lpc_filename: str, verbose: int = 0):
        """
        Function that synthesizes an acoustic signal from LPC features stored in a file, and save them in the same file
        with a .wav extension.
        """
        if verbose > 0:
            logger.info(f'Synthesizing {lpc_filename}.')

        try:
            out_filename = Path(lpc_filename).with_suffix(".wav").as_posix()
            lpc_coefficients = np.load(lpc_filename).astype(np.float32)
            net = LPCNet()
            result = np.hstack([net.synthesize(frame) for frame in lpc_coefficients])
            wavwrite(out_filename, 16000, result)
        except Exception as e:
            logger.error(f"Could not synthesize {lpc_filename} due to an unexpected exceptions: {str(e)}")
            return

        if verbose > 0:
            logger.info(f'Finished synthesizing {lpc_filename}.')

    def add_job(self, filename: str, verbose: int = 0):
        """
        Add a job to the queue which get synthesized as soon as capabilities are free on the cpu cores.
        """
        self.pool.apply_async(self._generate_audio_from_lpc, args=(filename, verbose))