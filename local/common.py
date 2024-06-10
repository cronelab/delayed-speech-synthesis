import h5py
import math
import numpy as np
import os
import scipy
from abc import ABC, abstractmethod
from sklearn.model_selection import LeaveOneOut
from scipy.ndimage import binary_dilation
from scipy.io import loadmat
from typing import Optional, Tuple, List, Dict, TypeAlias, Callable, Union
from functools import partial
from scipy.fftpack import dct


# region Subject specific
class SelectElectrodesFromBothGrids:
    """
    Feature selection covering the electrodes from both grids (ordered chan1, chan2, ..., chan64, chan65, chan66, ...)
    """
    def __init__(self):
        self.grid_mapping = [125, 123, 121, 119, 122, 111, 118, 124, 120, 126, 127, 116, 114, 113, 115, 117, 98, 97, 96,
                             104, 100, 102, 101, 99, 105, 112, 107, 106, 108, 103, 109, 110, 17, 21, 9, 28, 26, 31, 13,
                             27, 25, 22, 30, 11, 29, 23, 19, 15, 1, 2, 4, 0, 24, 12, 14, 7, 5, 18, 6, 10, 3, 8, 20, 16,
                             50, 33, 44, 51, 63, 40, 38, 46, 42, 48, 56, 37, 35, 41, 47, 58, 61, 60, 59, 43, 49, 45, 54,
                             62, 32, 53, 55, 52, 57, 39, 34, 36, 85, 84, 83, 87, 80, 86, 90, 78, 75, 92, 76, 88, 82, 94,
                             70, 74, 69, 66, 79, 71, 73, 77, 68, 67, 64, 65, 95, 93, 81, 72, 91, 89]

    def __len__(self):
        return len(self.grid_mapping)

    def __call__(self, data):
        return data[:, self.grid_mapping]


class SelectElectrodesOverSpeechAreas:
    """
    Feature selection covering only electrodes that I have been identified as carrying speech information. This
    includes the electrodes from the speech grid and four electrodes from the dorsal laryngeal area.
    Ordering: chan1, chan2, ...
    """
    def __init__(self):
        self.speech_grid_mapping = np.array([1, 2, 3, 0, 4, 11, 5, 6, 7, 10, 12, 9, 19, 8, 15, 20, 13, 14, 17, 22,
                                             18, 21, 29, 16, 23, 28, 35, 36, 27, 25, 26, 55, 45, 46, 44, 24, 37, 40,
                                             33, 34, 32, 51, 47, 39, 31, 54, 53, 30, 48, 38, 43, 41, 52, 61, 59, 62,
                                             49, 66, 60, 63, 58, 50, 42, 56, 67, 57, 81, 68]) + 1

        self.speech_grid_mapping = np.array([val for val in self.speech_grid_mapping if val not in [19, 38, 48, 52]])
        self.speech_grid_mapping -= 1
        self.speech_grid_mapping = np.array(sorted(self.speech_grid_mapping))

    def __len__(self):
        return len(self.speech_grid_mapping)

    def __call__(self, data):
        return data[:, self.speech_grid_mapping]

    def __repr__(self):
        return f"Channels: {', '.join(map(str, self.speech_grid_mapping + 1))}"


# Index array with respect to the brain plot figure to assign the electrode locations
img_layout = np.array([121, 122, 123, 113, 124, 125, 114, 115, 126, 105, 116, 127, 106, 117, 128, 107, 118,  97, 108,
                       119, 109,  98, 120,  99, 110,  89, 100, 111,  90, 101, 112,  91, 102,  81,  92, 103,  82,  93,
                       104,  83, 94,  73,  84,  95,  74,  85,  96,  75, 86,  65,  76,  87,  66,  77,  88,  67, 78,
                       68,  79,  69,  80,  70,  71,  72, 57,  58,  59,  60,  61,  62,  49,  63, 50,  64,  51,  52,
                       53,  54,  41,  55, 56,  42,  43,  44,  45,  46,  47,  33, 48,  34,  35,  36,  37,  38,  39,
                       25, 40,  26,  27,  28,  29,  30,  31,  17, 18,  32,  20,  19,  21,  22,  23,   9, 24,  10,
                       11,  12,  13,  14,  15,   1, 16,   2,   3,   4,   5,   6,   7,   8]) - 1
# endregion


# region Evaluation
class LeaveOneDayOut(LeaveOneOut):
    """
    Perform cross-validation on held-out recording sessions of a whole day.
    """
    def split(self, X, y=None, groups=None, start_with_day: Optional[str] = None):
        """
        Generate pairs of dates to distinguish days that will be used for training and the one that will be used for
        testing
        :param X: List of strings in the form "year_month_day" that indicate the recording days
        :param y: Not used, just here for compatability
        :param groups: Not used, just here for compatability
        :param start_with_day: If not None will rotate the list of days so that start_with_day will be first test day
        :return: Tuple containing one list of the days used for training and one string that indicates the testing day.
        """
        ordered_days = sorted(X)
        if start_with_day is not None:
            if start_with_day not in ordered_days:
                raise ValueError(f"The day {start_with_day} is not in the list of provided days {ordered_days}.")

            # Rotate sorted list to re-arrange days that test day will be the first day in the list
            while ordered_days[0] != start_with_day:
                ordered_days.append(ordered_days.pop(0))

        indices = np.arange(len(ordered_days))
        for test_index in self._iter_test_masks(indices, None, None):
            train_index = indices[np.logical_not(test_index)]
            train_day = [ordered_days[i] for i in train_index]
            test_day = ordered_days[np.argmax(test_index)]
            yield train_day, test_day
# endregion


# region VoiceActivityDetection (VAD)
class VoiceActivityDetectionSmoothing:
    """
    Class with considers each VAD label with respect to its neighbors (both past and future) to correct potential
    misclassification from the neural VAD model. Based on the number of context frames, which will be applied on both
    sides of each frame, this class will introduce a delay which is proportional to the number of future frames. In
    order to keep the alignment between speech and neural data, this class uses a ringbuffer and outputs frames only at
    times when it can make a reliable prediction on the whole window.
    """
    def __init__(self, nb_features: int, context_frames: int, proportion_threshold: float = 0.6, shift: float = 0.01):
        self.frameshift = shift
        self.nb_features = nb_features
        self.vad_context_frames = context_frames
        self.vad_proportion_threshold = proportion_threshold
        self.buffer_size = 2 * self.vad_context_frames + 1
        self.buffer = np.zeros((self.buffer_size, self.nb_features), dtype=np.float32)
        self.labels = np.zeros(self.buffer_size, dtype=bool)
        self.write_pointer = 2 * self.vad_context_frames
        self.read_pointer = 0

    def insert(self, data: np.ndarray, speech_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        resulting_label = np.zeros(len(speech_labels), dtype=bool)
        resulting_data = np.zeros((len(speech_labels), self.nb_features), dtype=np.float32)

        for i in range(len(speech_labels)):
            # Insert label into ringbuffer
            label = speech_labels[i]
            self.labels[self.write_pointer] = label

            # Insert high-gamma frame into ringbuffer
            frame = data[i]
            self.buffer[self.write_pointer, :] = frame

            # Compute output label based on the current window
            ratio = np.count_nonzero(self.labels) / self.buffer_size
            output_label = True if ratio >= self.vad_proportion_threshold else False
            resulting_label[i] = output_label
            resulting_data[i, :] = self.buffer[self.read_pointer, :]

            # Advance both read and write pointer
            self.write_pointer = (self.write_pointer + 1) % self.buffer_size
            self.read_pointer = (self.read_pointer + 1) % self.buffer_size

        return resulting_data, resulting_label

    def __repr__(self):
        return f"VAD Smoothing(Window size: {self.buffer_size * self.frameshift:.02f} s " \
               f"(introduced delay: {math.floor(self.buffer_size / 2) * self.frameshift} s), " \
               f"requires {self.vad_proportion_threshold * 100:.01f}% of frames to be speech)"


class SpeechSegmentHistory:
    """
    Class which stores frames identified as speech in a ringbuffer and returns the complete speech segment as soon as
    the voice activity detection reports that the speech segment has ended.
    """
    def __init__(self, nb_features: int, buffer_size: int, context: int = 0):
        self.buffer = np.zeros((buffer_size, nb_features), dtype=np.float32)
        self.write_pointer = 0
        self.context = context
        self.speech_frame_counter = 0
        self.future_frame_counter = 0

    @staticmethod
    def _get_positions(read_pointer: int, write_pointer: int, buffer_size: int) -> List[int]:
        """
        Return an array of indices that specifies which elements in the array are being selected. Takes care of the
        buffer size and wraps around the end of the ringbuffer while remaining the order.
        """
        result = []
        while read_pointer != write_pointer:
            result.append(read_pointer)
            read_pointer = (read_pointer + 1) % buffer_size

        return result

    def insert(self, data: np.ndarray, speech_labels: np.ndarray) -> List[np.ndarray]:
        """
        Insert each frame into a ringbuffer and return a full speech segment after patient finished with speaking.
        :param data: Array of high-gamma frames
        :param speech_labels: Array of voice activity labels
        :return: List of speech segments that have been marked as completed, otherwise empty list
        """
        result = []
        for index in range(len(speech_labels)):
            frame = data[index, :]
            label = speech_labels[index]

            # Insert frame in ringbuffer
            self.buffer[self.write_pointer, :] = frame
            self.write_pointer = (self.write_pointer + 1) % len(self.buffer)

            # If we have seen a speech frame, increase the internal counter
            if label:
                self.speech_frame_counter += 1

            if not label and self.speech_frame_counter > 0:
                self.future_frame_counter += 1

                if self.future_frame_counter >= self.context:
                    stop = self.write_pointer if self.context > 0 else (self.write_pointer - 1) % len(self.buffer)
                    start = (stop - 2 * self.context - self.speech_frame_counter) % len(self.buffer)

                    positions = self._get_positions(start, stop, buffer_size=len(self.buffer))
                    result.append(self.buffer[positions])

                    # Reset counters
                    self.speech_frame_counter = 0
                    self.future_frame_counter = 0

        return result
# endregion


# region Preprocessing
class BadChannelCorrection:
    """
    Replace content of predefined bad channels with the mean from neighboring (non-bad) channels.
    """
    def __init__(self, bad_channels: List[int], grids: List[np.ndarray], layout: np.ndarray):
        """
        :param bad_channels: List of integer values (1 to 128) indicating which channels have been marked as
        bad channels
        :param grids: List of 2D numpy arrays that specify the numbering in each grid
        :param layout: List of integers that map each column in the data array to the channel in the grids list.
                       Example: [chan52, ...]
        """
        self.grids = grids
        self.layout = layout
        self.masks = [np.ones(grid.shape, dtype=bool) for grid in grids]
        self._construct_masks(bad_channels=bad_channels)
        self.footprint = self._get_footprint()
        self.patches = [(bad_channel, self._identify_neighbors(bad_channel)) for bad_channel in bad_channels]

        # Patches will store a tuple of the index where the bad channel is located in the data (according to the layout
        # information), and a list of indices from where in the data the mean will be computed.
        self.patches = [(np.where(self.layout == bad_channel)[0], self._find_neighbors_idx(neighbors))
                        for bad_channel, neighbors in self.patches]

    def _get_footprint(self):
        """
        Uses by default an 8-neighbor footprint. Override this function in case for a different neighboring set.
        """
        footprint = np.ones(9, dtype=bool).reshape((3, 3))
        footprint[1, 1] = False
        return footprint

    def _find_neighbors_idx(self, neighbors: List[int]) -> np.ndarray:
        return np.concatenate([np.where(self.layout == neighbor)[0] for neighbor in neighbors])

    def _identify_grid_index(self, channel: int) -> int:
        """
        Returns the index in which grid the channel was found.
        """
        for i, grid in enumerate(self.grids):
            if channel in grid:
                return i

        raise IndexError('Channel could not be found in given grids.')

    def _construct_masks(self, bad_channels: List[int]) -> None:
        """
        Construct binary masks for each grid that would reject all bad channels in that grid.
        """
        for bad_channel in bad_channels:
            grid_index = self._identify_grid_index(bad_channel)
            grid = self.grids[grid_index]
            row, col = np.where(grid == bad_channel)
            self.masks[grid_index][row, col] = False

    def _identify_neighbors(self, channel: int) -> List[int]:
        """
        Returns a list of all neighboring channels which should be considered for calculating the mean.
        """
        grid_index = self._identify_grid_index(channel)
        grid = self.grids[grid_index]
        row, col = np.where(grid == channel)
        mask = np.zeros(grid.shape, dtype=bool)
        mask[row, col] = True
        mask = binary_dilation(mask, structure=self.footprint)
        mask = mask & self.masks[grid_index]
        return grid[mask]

    def __call__(self, data: np.ndarray) -> np.ndarray:
        result = data.copy()
        for bad_channel_location, neighbors in self.patches:
            result[:, bad_channel_location] = np.mean(data[:, neighbors], axis=1).reshape((len(data), -1))

        return result

    def __len__(self):
        return len(self.patches)

    def __repr__(self):
        # Correcting 4 bad channels: 38 -> [37, 39, ...], ...
        items = []
        for bc_index, neighbor_indices in self.patches:
            bc = self.layout[bc_index].item()
            neighbors = [self.layout[neighbor_index] for neighbor_index in neighbor_indices]
            items.append(f"{bc} -> {str(neighbors)}")
        return f"Correcting {len(self.patches)} bad channels: {', '.join(items)}"


class CommonAverageReferencing:
    """
    Subtract the global average within each electrode grid at each point in time from each electrode channel in that
    particular grid. Specified bad channels will not be included in computing the mean.
    Expects data to be in the form: T x E.
    """
    def __init__(self, exclude_channels: List[int], grids: List[np.ndarray], layout: np.ndarray):
        """
        :param exclude_channels: List of integer values (1 to 128) indicating which channels have been marked to be
        excluded from being used to be included in computing the global mean (e.g. bad channels)
        :param grids: List of 2D numpy arrays that specify the electrode alignment in each grid
        :param layout: List of integers that map each column in the data array to the channel in the grids list.
                       Example: [52, ...] if data has the 52nd channel in the first position.
        """
        self.grids = grids
        self.layout = layout

        # Construct the selection masks on where in the data to apply the CAR filtering.
        self.selection_masks_application = [np.isin(layout, grid) for grid in grids]

        # Construct the selection masks on where in the data to compute the global mean.
        self.selection_masks_computation = []
        for grid, mask_appl in zip(self.grids, self.selection_masks_application):
            mask_comp = mask_appl.copy()
            for excluded_channel in exclude_channels:
                if excluded_channel in grid:
                    mask_comp[np.argmax(layout == excluded_channel)] = False

            self.selection_masks_computation.append(mask_comp)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        result = data.copy()
        for mask_comp, mask_appl in zip(self.selection_masks_computation, self.selection_masks_application):
            means_per_time_point = np.mean(data[:, mask_comp], axis=1).reshape((-1, 1))
            means_per_time_point = np.tile(means_per_time_point, reps=(1, np.count_nonzero(mask_appl)))
            result[:, mask_appl] = result[:, mask_appl] - means_per_time_point

        return result

    def __repr__(self):
        """
        CommonAverageReferencing: 2 grids
        Grid 0:
            mask_appl: [...]
            mask_comp: [...]

        Grid 1:
            mask_appl: [...]
            mask_comp: [...]
        """
        string_info = f"CommonAverageReferencing ({len(self.grids)} grids):\n"
        for i, (m_appl, m_comp) in enumerate(zip(self.selection_masks_application, self.selection_masks_computation)):
            string_info += f"Grid {i}\n"
            string_info += f"\tmask_appl: [{', '.join(map(str, self.layout[m_appl]))}]\n"
            string_info += f"\tmask_comp: [{', '.join(map(str, self.layout[m_comp]))}]\n"

        return string_info


class ZScoreNormalization:
    """
    Normalizes each channel according to predefined means and standard deviations.
    """
    def __init__(self, channel_means: np.ndarray, channel_stds: np.ndarray):
        self.channel_means = channel_means
        self.channel_stds = channel_stds

    def __call__(self, data):
        return (data - self.channel_means) / self.channel_stds


def save_data_to_hdf(filename: str, parameters: Dict[str, np.ndarray], overwrite: bool = False) -> bool:
    """
    Store timed aligned neural and acoustic data to .hdf container.
    """
    if os.path.exists(filename) and not overwrite:
        print(f'File {filename} already exists and overwrite is set to False. Training data is not stored.')
        return False

    with h5py.File(filename, 'w') as hf:
        for container_name, data in parameters.items():
            hf.create_dataset(container_name, data=data)

    return True
# endregion


# region BCI2000 wrapper
TrialIndices: TypeAlias = Tuple[str, int, int]


class BCI2000MatFile:
    """
    Wrapper class which makes the contents from the BCI2000 mat files more accessible.
    """
    def __init__(self, mat_filename: str):
        self.mat_filename = mat_filename
        self.mat = loadmat(self.mat_filename, simplify_cells=True)
        self.fs = self.mat['parameters']['SamplingRate']['NumericValue']

    def bad_channels(self) -> Optional[List[int]]:
        if 'bad_channels' in self.mat.keys():
            bad_channels = self.mat['bad_channels']
            if type(bad_channels) is np.ndarray:
                bad_channels = bad_channels.tolist()

            bad_channels = [bad_channels] if type(bad_channels) is not list else bad_channels
            bad_channels = [int(bad_channel[4:]) for bad_channel in bad_channels]
        else:
            bad_channels = None

        return bad_channels

    def contaminated_channels(self) -> Optional[List[int]]:
        if "contaminated_electrodes" in self.mat.keys():
            contaminated_electrodes = self.mat['contaminated_electrodes']
            if type(contaminated_electrodes) is int:
                contaminated_electrodes = [contaminated_electrodes, ]
            else:
                contaminated_electrodes = contaminated_electrodes.tolist()
            return contaminated_electrodes
        else:
            return None

    def trial_indices(self, min_trial_length: Optional[float] = None) -> List[TrialIndices]:
        stimuli = ExperimentMapping.extract_stimuli_values(self.mat)

        # Read stimulus code
        stimulus_code = self.mat['states']['StimulusCode']
        experiment_class = ExperimentMapping.get_experiment_class(mat_filename=self.mat_filename)
        experiment = experiment_class(stimulus_code, stimuli)
        trial_indices = experiment.get_trial_indices()

        if min_trial_length is not None:
            nb_min_samples = min_trial_length * self.fs
            trial_indices = [(label, start, max(stop, start + nb_min_samples)) for label, start, stop in trial_indices]

        return trial_indices

    def stimuli_indices(self) -> List[TrialIndices]:
        stimuli = ExperimentMapping.extract_stimuli_values(self.mat)

        # Read stimulus code
        stimulus_code = self.mat['states']['StimulusCode']
        experiment_class = ExperimentMapping.get_experiment_class(mat_filename=self.mat_filename)
        experiment = experiment_class(stimulus_code, stimuli)
        stimuli_indices = experiment.get_stimuli_indices()

        return stimuli_indices

    def signals(self) -> np.ndarray:
        signals = self.mat['signal']
        gain = self.mat['parameters']['SourceChGain']['NumericValue']
        return signals * gain

    def ordered_stimulus_codes(self) -> List[int]:
        stimulus_code = self.mat['states']['StimulusCode']

        # Get stimulus codes
        stimulus_codes = np.unique(stimulus_code).tolist()
        stimulus_codes = sorted(stimulus_codes)[1:]

        return stimulus_codes
# endregion


# region Voice Activity Detection
class MelFilterBank:
    def __init__(self, specSize, numCoefficients, sampleRate):
        numBands = int(numCoefficients)

        # Set up center frequencies
        minMel = 0
        maxMel = self.freqToMel(sampleRate / 2.0)
        melStep = (maxMel - minMel) / (numBands + 1)

        melFilterEdges = np.arange(0, numBands + 2) * melStep

        # Convert center frequencies to indices in spectrum
        centerIndices = list(
            map(lambda x: self.freqToBin(math.floor(self.melToFreq(x)), sampleRate, specSize), melFilterEdges))

        # Prepare matrix
        filterMatrix = np.zeros((numBands, specSize))

        # Construct matrix with triangular filters
        for i in range(numBands):
            start, center, end = centerIndices[i:i + 3]
            k1 = float(center - start)
            k2 = float(end - center)
            up = (np.array(range(start, center)) - start) / k1
            down = (end - np.array(range(center, end))) / k2

            filterMatrix[i][start:center] = up
            filterMatrix[i][center:end] = down

        # Save matrix and its best-effort inverse
        self.melMatrix = filterMatrix.transpose()
        self.melMatrix = self.makeNormal(self.melMatrix / self.normSum(self.melMatrix))

        self.melInvMatrix = self.melMatrix.transpose()
        self.melInvMatrix = self.makeNormal(self.melInvMatrix / self.normSum(self.melInvMatrix))

    def normSum(self, x):
        retSum = np.sum(x, axis=0)
        retSum[np.where(retSum == 0)] = 1.0
        return retSum

    def fuzz(self, x):
        return x + 0.0000001

    def freqToBin(self, freq, sampleRate, specSize):
        return int(math.floor((freq / (sampleRate / 2.0)) * specSize))

    def freqToMel(self, freq):
        return 2595.0 * math.log10(1.0 + freq / 700.0)

    def melToFreq(self, mel):
        return 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0)

    def toMelScale(self, spectrogram):
        return (np.dot(spectrogram, self.melMatrix))

    def fromMelScale(self, melSpectrogram):
        return (np.dot(melSpectrogram, self.melInvMatrix))

    def makeNormal(self, x):
        nanIdx = np.isnan(x)
        x[nanIdx] = 0

        infIdx = np.isinf(x)
        x[infIdx] = 0

        return (x)

    def toMels(self, spectrogram):
        return (self.toMelScale(spectrogram))

    def fromMels(self, melSpectrogram):
        return (self.fromMelScale(melSpectrogram))

    def toLogMels(self, spectrogram):
        return (self.makeNormal(np.log(self.fuzz(self.toMelScale(spectrogram)))))

    def fromLogMels(self, melSpectrogram):
        return (self.makeNormal(self.fromMelScale(np.exp(melSpectrogram))))


class EnergyBasedVad:
    """
    Energy based VAD computation. Should be equal to compute-vad from Kaldi.

    Arguments:
        log_mels (numpy array): log-Mels which get transformed into MFCCs
        mfcc_coeff (int): Number of MFCC coefficients (exclusive first one)
        energy_threshold (float): If this is set to s, to get the actual threshold we let m be
            the mean log-energy of the file, and use s*m + vad-energy-threshold (float, default = 0.5)
        energy_mean_scale (float): Constant term in energy threshold for MFCC0 for VAD
            (also see --vad-energy-mean-scale) (float, default = 5)
        frames_context (int): Number of frames of context on each side of central frame,
            in window for which energy is monitored (int, default = 0)
        proportion_threshold (float): Parameter controlling the proportion of frames within
            the window that need to have more energy than the threshold (float, default = 0.6)
        export_to_file (str): filename to export the VAD in .lab file format (readable with audacity)
    """
    def __init__(self, energy_threshold=4, energy_mean_scale=1, frames_context=5, proportion_threshold=0.6):
        self.vad_energy_threshold = energy_threshold
        self.vad_energy_mean_scale = energy_mean_scale
        self.vad_frames_context = frames_context
        self.vad_proportion_threshold = proportion_threshold
        self.mfcc_coeff = 13
        self.frame_shift = 0.01
        self.window_length = 0.05

    def from_wav(self, wav, sampling_rate=16000):
        # segment audio into windows
        window_size = int(sampling_rate * self.window_length)
        window_shift = int(sampling_rate * self.frame_shift)
        nb_windows = math.floor((len(wav) - window_size) / window_shift) + 1

        audio_segments = np.zeros((nb_windows, window_size))
        for win in range(nb_windows):
            start_audio = int(round(win * window_shift))
            stop_audio = int(round(start_audio + window_size))

            audio_segment = wav[start_audio:stop_audio]
            audio_segments[win, :] = audio_segment

        # create spectrogram from wav
        spectrogram = np.zeros((audio_segments.shape[0], window_size // 2 + 1), dtype='complex')

        win = scipy.hanning(window_size)
        for w in range(nb_windows):
            a = audio_segments[w, :] / (2 ** 15)
            spec = np.fft.rfft(win * a)
            spectrogram[w, :] = spec

        mfb = MelFilterBank(spectrogram.shape[1], 40, sampling_rate)
        log_mels = (mfb.toLogMels(np.abs(spectrogram)))

        return self.from_log_mels(log_mels=log_mels)

    def from_log_mels(self, log_mels):
        self.mfccs = dct(log_mels)
        self.mfccs = self.mfccs[:, 0:self.mfcc_coeff + 2]

        return self.from_mfccs(self.mfccs)

    def from_mfccs(self, mfccs):
        self.mfccs = mfccs
        vad = self._compute_vad()
        return vad

    def _compute_vad(self):
        # VAD computation
        log_energy = self.mfccs[:, 0]
        output_voiced = np.empty(len(self.mfccs), dtype=bool)

        energy_threshold = self.vad_energy_threshold
        if self.vad_energy_mean_scale != 0:
            assert self.vad_energy_mean_scale > 0
            energy_threshold += self.vad_energy_mean_scale * np.sum(log_energy) / len(log_energy)

        assert self.vad_frames_context >= 0
        assert 0.0 < self.vad_proportion_threshold < 1

        for frame_idx in range(0, len(self.mfccs)):
            num_count = 0.0
            den_count = 0.0

            for t2 in range(frame_idx - self.vad_frames_context, frame_idx + self.vad_frames_context):
                if 0 <= t2 < len(self.mfccs):
                    den_count += 1
                    if log_energy[t2] > energy_threshold:
                        num_count += 1

            if num_count >= den_count * self.vad_proportion_threshold:
                output_voiced[frame_idx] = True
            else:
                output_voiced[frame_idx] = False

        return output_voiced

    def convert_vad_to_lab(self, filename, vad):
        last_i = None
        s = None
        r = ''

        for t, i in enumerate(vad):
            if last_i is None:
                last_i = i
                s = 0

            if i != last_i:
                e = t * self.frame_shift  # 10 ms
                r += '{:.2f}\t{:.2f}\t{}\n'.format(s, e, int(last_i))

                s = t * 0.01
                last_i = i

        r += '{:.2f}\t{:.2f}\t{}\n'.format(s, len(vad) * self.frame_shift, int(last_i))

        with open(filename, 'w+') as f:
            f.write(r)
# endregion


# region Experiments
class Experiment(ABC):
    """
    Abstract class defining the interface for extracting trial segments from different experiment tasks.
    """
    def __init__(self, stimulus_code: np.ndarray, stimuli: Union[Dict[int, str], List[str]]):
        self.stimulus_code = stimulus_code
        self.stimuli = stimuli

        # Infer stimuli dict from the positions in the list, starting at index 1 to reserve 0 for not being a stimuli.
        if isinstance(self.stimuli, list):
            self.stimuli = {(index + 1): item for index, item in enumerate(self.stimuli)}

    def __repr__(self):
        return f'{self.__class__.__name__}(len: {len(self.stimulus_code)} samples, with {len(self.stimuli)} stimuli ' \
               f'across {len(self.get_trial_indices())} trials)'

    def _determine_trial_boundaries(self) -> List[Tuple[int, int]]:
        diff = np.where(self.stimulus_code[:-1] != self.stimulus_code[1:])[0] + 1
        return list(zip(diff[::], diff[1::]))

    @staticmethod
    def trial_indices_to_lab(filename: str, trial_indices: List[Tuple[str, int, int]], fs: int):
        with open(filename, 'w') as f:
            for label, start, stop in trial_indices:
                f.write(f'{start / fs:.03f}\t{stop / fs:.03f}\t{label}\n')

    @abstractmethod
    def get_trial_indices(self) -> List[Tuple[str, int, int]]:
        """
        Abstract method which return a list of tuples containing the label name, start and stop indices of each trial.
        """
        ...

    @abstractmethod
    def get_stimuli_indices(self) -> List[Tuple[str, int, int]]:
        """
        Abstract method which return a list of tuples containing the label name, start and stop indices of each
        stimulus presentation phase before a trial.
        """
        ...

    def get_webfm_baseline_windows(self, fs: int, length: float = 0.8) -> List[Tuple[str, int, int]]:
        """
        Method which return a list of tuples that contain 0.8 seconds pre-stimulus cue start to compute
        high-gamma baseline on.
        """
        trials = self.get_stimuli_indices()
        baseline_windows = [('BL', int(start - length * fs), start) for _, start, _ in trials]
        return baseline_windows

    def get_experiment_run_indices(self) -> Tuple[str, int, int]:
        """
        Returns a tuple of start and stop indices of the complete experiment run. This gets determined by the stimulus
        codes, since each recording could have data before and after the actual run.
        """
        trial_boundaries = self._determine_trial_boundaries()
        start = trial_boundaries[0][0]

        trials = self.get_trial_indices()
        stop = trials[-1][2]

        return 'Experiment run', start, stop

    @staticmethod
    def get_stimuli_values() -> Optional[list]:
        return None


class SyllableRepetition(Experiment):
    """
    Task in which syllables are audibly presented and after an acoustic hint the patient repeats the presented syllable.
    """
    @staticmethod
    def _swap_auditory_stimuli_codes(stimuli_codes: np.ndarray, trials: List[Tuple[int, int]]) -> np.ndarray:
        stimuli_presentation_segments = trials[::2]
        patient_speaking_segments = trials[1::2]

        new_stimuli_codes = stimuli_codes.copy()

        # Transfer stimuli codes from presentation to actual speaking
        for k, (start, stop) in enumerate(patient_speaking_segments):
            code = stimuli_codes[stimuli_presentation_segments[k][0]]
            new_stimuli_codes[start:stop] = code

        # Zero out stimuli codes from presentation
        for start, stop in stimuli_presentation_segments:
            new_stimuli_codes[start:stop] = 0

        return new_stimuli_codes

    @staticmethod
    def _determine_trial_length(trials: List[Tuple[int, int]]) -> int:
        start, stop = trials[1]
        return stop - start

    def get_trial_indices(self) -> List[Tuple[str, int, int]]:
        """
        Stimulus codes != 0 indicate the segments in which the syllable is acoustically presented. Afterwards, the
        stimulus code switches to 0 while the patient repeats the syllable.

        :return: List of tuples in which the first component represents the label of the stimuli that was used during
        acoustic presentation, followed by start and end indices of that segment.
        """
        trials = self._determine_trial_boundaries()

        # Append last trial which cannot identified through the difference method.
        trial_length = self._determine_trial_length(trials)
        trial_length = min(trial_length, len(self.stimulus_code))
        trials.append((trials[-1][1], trials[-1][1] + trial_length))

        stim_codes = self._swap_auditory_stimuli_codes(self.stimulus_code, trials)
        trials = [(self.stimuli[stim_codes[start]], start, stop) for (start, stop) in trials if stim_codes[start] != 0]
        return trials

    def _stimuli_extraction(self, entry_condition: Callable, exit_condition: Callable) -> List[Tuple[str, int, int]]:
        start = None
        label = None
        result = []
        for i in range(len(self.stimulus_code)):
            if entry_condition(self.stimulus_code[i]) and start is None:
                start = i
                label = self.stimuli[self.stimulus_code[i]]

            if exit_condition(self.stimulus_code[i]) and start is not None:
                result.append((label, start, i))
                start = None
                label = None

        return result

    def get_stimuli_indices(self) -> List[Tuple[str, int, int]]:
        entry_condition = partial(lambda stimulus_code: stimulus_code != 0)
        exit_condition = partial(lambda stimulus_code: stimulus_code == 0)
        return self._stimuli_extraction(entry_condition=entry_condition, exit_condition=exit_condition)


class KeywordReading(Experiment):
    """
    Patient reading the keywords. Each word is shown for a few seconds on screen. Word presented for 1s.
    Fixation (Stimulus_Code 1) lasts 1.5s, ISI 1s. Entire duration 4.5s.
    For this experiment, trial and stimuli indices are the same.
    """
    def _stimuli_extraction(self, entry_condition: Callable, exit_condition: Callable) -> List[Tuple[str, int, int]]:
        start = None
        label = None
        result = []
        for i in range(len(self.stimulus_code)):
            if entry_condition(self.stimulus_code[i]) and start is None:
                start = i
                label = self.stimuli[self.stimulus_code[i]]

            if exit_condition(self.stimulus_code[i]) and start is not None:
                result.append((label, start, i))
                start = None
                label = None

        return result

    def get_trial_indices(self) -> List[Tuple[str, int, int]]:
        return self.get_stimuli_indices()

    def get_stimuli_indices(self) -> List[Tuple[str, int, int]]:
        entry_condition = partial(lambda stimulus_code: stimulus_code != 0)
        exit_condition = partial(lambda stimulus_code: stimulus_code == 0)
        return self._stimuli_extraction(entry_condition=entry_condition, exit_condition=exit_condition)


class ExperimentMapping(dict):
    """
    Map experiment names to Experiment class for extracting trial indices.
    """
    def __init__(self):
        super().__init__()
        mapping = {
            'SyllableRepetition': SyllableRepetition,
            'KeywordReading': KeywordReading,
            'KeywordSynthesis': KeywordReading,
        }
        self.update(mapping)

    @staticmethod
    def get_experiment_class(mat_filename: str) -> Optional[Experiment]:
        """
        Based on the filename, return the appropriate experiment class which can be used to extract trial indices.
        """
        filename = os.path.basename(mat_filename)
        mapping = ExperimentMapping()

        for key in mapping.keys():
            if key in filename:
                return mapping[key]

        return None

    @staticmethod
    def extract_stimuli_values(mat: dict) -> List[str]:
        """
        Helper function to extract stimuli values from a loaded .mat file. This function takes care that a list of
        stimuli values is returned.
        """
        stimuli = mat['parameters']['Stimuli']['Value']
        if stimuli.ndim == 1:
            return [stimuli[0]]
        else:
            return stimuli[0].tolist()
# endregion
