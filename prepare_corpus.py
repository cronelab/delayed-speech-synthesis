import argparse
import numpy as np
import os
import logging
from typing import Optional, List, Dict
from pydub import AudioSegment, effects
from pathlib import Path
from tqdm import tqdm
from scipy.io.wavfile import read as wavread
from LPCNet import LPCFeatureEncoder
from local.common import BCI2000MatFile, save_data_to_hdf
from local.common import CommonAverageReferencing, BadChannelCorrection, SelectElectrodesFromBothGrids
from local.common import EnergyBasedVad, ExperimentMapping
from local.units import HighGammaExtractor
import matplotlib.pyplot as plt


logger = logging.getLogger("prepare_corpus.py")


class FeatureExtractionPipeline:
    """
    Extract features and targets, such as LPC coefficients and voice activity labels, from all provided mat files.
    """
    def __init__(self, mat_filename: Path, wav_filename: Path, min_trial_length: Optional[float] = None):
        self.mat_filename = mat_filename
        self.wav_filename = wav_filename
        self.min_trial_length = min_trial_length
        self.mat = BCI2000MatFile(mat_filename=self.mat_filename.as_posix())
        self.fs_audio, self.wav = wavread(wav_filename)

    @staticmethod
    def _normalize_audio(audio: np.ndarray, fs: int, normalization_factor: float = -3.0) -> np.ndarray:
        """
        Normalize the extracted audio segment to a certain dB level.
        """
        segment = AudioSegment(audio.tobytes(), frame_rate=fs, sample_width=audio.dtype.itemsize, channels=1)
        segment = effects.normalize(segment)
        segment = segment.apply_gain(normalization_factor)
        return np.array(segment.get_array_of_samples())

    def get_features(self, show_pbar: bool = False) -> np.ndarray:
        features = []
        ecog = self.mat.signals()
        desc = f"Computing high-gamma features for {self.mat_filename.name}"
        pbar = not show_pbar
        for _, start, stop in tqdm(self.mat.trial_indices(self.min_trial_length), desc=desc, disable=pbar):
            extractor = get_feature_extractor(self.mat)
            feats = extractor.extract_features(ecog[start:int(stop + (0.04 * self.mat.fs)), :])
            features.append(feats)

        return np.concatenate(features)

    def get_lpc_coefficients(self, norm: float = -3.0, show_pbar: bool = False) -> np.ndarray:
        lpc_features = []
        desc = f"Computing LPC features for {self.wav_filename.name}"
        pbar = not show_pbar
        for label, start, stop in tqdm(self.mat.trial_indices(self.min_trial_length), desc=desc, disable=pbar):
            # Convert start and stop indices to high-fidelity audio sampling rate
            start = int(start * self.fs_audio / self.mat.fs)
            stop = int(stop * self.fs_audio / self.mat.fs) + int(0.04 * self.fs_audio)

            trial_audio = self.wav[start:stop]
            if not (label == "SILENCE"):
                trial_audio = self._normalize_audio(trial_audio, fs=self.fs_audio, normalization_factor=norm)

            # Shift audio by 16 ms to account for filter delay
            filter_delay_pad = np.zeros(int(0.016 * self.fs_audio), dtype=np.int16)
            trial_audio = np.hstack([filter_delay_pad, trial_audio[:-len(filter_delay_pad)]])

            # Compute LPC features
            encoder = LPCFeatureEncoder()
            features = encoder.compute_LPC_features(trial_audio)
            lpc_features.append(features[3:-1])

        return np.concatenate(lpc_features)

    def get_vad_labels(self, norm: float = -3.0, show_pbar: bool = False) -> np.ndarray:
        vad_labels = []
        desc = f"Computing VAD labels for {self.wav_filename.name}"
        pbar = not show_pbar
        for i, (label, start, stop) in enumerate(tqdm(self.mat.trial_indices(self.min_trial_length), desc=desc, disable=pbar)):
            # Convert start and stop indices to high-fidelity audio sampling rate
            start = int(start * self.fs_audio / self.mat.fs)
            stop = int(stop * self.fs_audio / self.mat.fs) + int(0.04 * self.fs_audio)

            trial_audio = self.wav[start:stop]
            if not (label == "SILENCE"):
                trial_audio = self._normalize_audio(trial_audio, fs=self.fs_audio, normalization_factor=norm)

            # Shift audio by 16 ms to account for filter delay
            filter_delay_pad = np.zeros(int(0.016 * self.fs_audio), dtype=np.int16)
            trial_audio = np.hstack([filter_delay_pad, trial_audio[:-len(filter_delay_pad)]])

            # Compute LPC features
            vad = EnergyBasedVad()
            labels = vad.from_wav(trial_audio, sampling_rate=self.fs_audio)

            if label == "SILENCE":
                labels = np.zeros_like(labels)
            vad_labels.append(labels)

            # Plot VAD labels and the acoustic speech signal for error checking
            fig, ax = plt.subplots(1, 1, num=1, clear=True)
            xs = np.linspace(0, len(trial_audio) / self.fs_audio, len(trial_audio))
            ax.plot(xs, trial_audio / np.max(trial_audio), c="blue")
            xs = np.linspace(0, len(trial_audio) / self.fs_audio, len(labels))
            ax.plot(xs, labels, c="orange")

            mat_filename_path = Path(self.mat.mat_filename)
            vad_plot_filename = os.path.join("/tmp/vad_labels", mat_filename_path.parent.name,
                                             f"{mat_filename_path.with_suffix('').name}_{i + 1:03d}.png")
            os.makedirs(Path(vad_plot_filename).parent, exist_ok=True)
            plt.savefig(vad_plot_filename, dpi=72)

        return np.concatenate(vad_labels)

    def get_trial_ids(self) -> np.ndarray:
        trial_ids = list()
        stimuli = ExperimentMapping.extract_stimuli_values(self.mat.mat)

        last_stimuli_code = None
        for label, start, stop in self.mat.trial_indices(self.min_trial_length):
            interval = int(stop + (0.04 * self.mat.fs)) - start
            overlap = 0.04 * self.mat.fs
            window_shift = 0.01 * self.mat.fs
            num_windows = int(np.floor((interval - overlap) / window_shift))

            stimuli_code = stimuli.index(label) + 1
            if last_stimuli_code is None or last_stimuli_code != stimuli_code:
                trial_ids.append(np.ones(num_windows) * stimuli_code)
                last_stimuli_code = stimuli_code
            else:
                trial_ids.append(np.ones(num_windows) * stimuli_code * -1)
                last_stimuli_code = stimuli_code * -1

        return np.hstack(trial_ids).astype(np.int16)

    def accumulative_audio_duration(self) -> float:
        accumulative_sum = 0.0
        for _, start, stop in self.mat.trial_indices(self.min_trial_length):
            accumulative_sum += stop - start

        return accumulative_sum / self.mat.fs


def get_feature_extractor(cleaned_mat_file: BCI2000MatFile) -> HighGammaExtractor:
    fs = cleaned_mat_file.fs
    bad_channels = cleaned_mat_file.bad_channels()
    contaminated_channels = cleaned_mat_file.contaminated_channels()

    # Reorder and select only channels from both grids
    feature_selection = SelectElectrodesFromBothGrids()
    pre_transforms = [feature_selection]

    speech_grid = np.flip(np.arange(64, dtype=np.int16).reshape((8, 8)) + 1, axis=0)
    motor_grid = np.flip(np.arange(64, dtype=np.int16).reshape((8, 8)) + 65, axis=0)
    layout = np.arange(128) + 1
    car = CommonAverageReferencing(exclude_channels=[19, 38, 48, 52], grids=[speech_grid, motor_grid], layout=layout)
    pre_transforms.append(car)
    post_transforms = None

    # Initialize channel correction
    if contaminated_channels is not None:
        logger.debug(f"Found contaminated channels in {cleaned_mat_file.mat_filename}: {contaminated_channels}.")
        corrected_channels = bad_channels + contaminated_channels
        ch_correction = BadChannelCorrection(bad_channels=corrected_channels, grids=[speech_grid, motor_grid],
                                             layout=layout)
        post_transforms = [ch_correction, ]

    # Initialize HighGammaExtraction module
    nb_electrodes = len(feature_selection)
    ex = HighGammaExtractor(fs=fs, nb_electrodes=nb_electrodes, pre_transforms=pre_transforms,
                            post_transforms=post_transforms)

    return ex


class ZScoresFromSyllableRepetitions(dict):
    """
    Creates a dictionary that stores z-score normalization statistics computed from the syllable repetition recordings.
    """
    def __init__(self, syllable_recordings: Dict[str, Path], show_pbar: bool = False):
        super(ZScoresFromSyllableRepetitions, self).__init__()

        desc = "Computing z-score statistics per day on SyllableRepetition data"
        pbar = not show_pbar
        for day, syllable_recording_path in tqdm(syllable_recordings.items(), desc=desc, disable=pbar):
            syllable_recording = BCI2000MatFile(mat_filename=syllable_recording_path.as_posix())

            ecog = syllable_recording.signals()
            data = list()
            for _, start, stop in syllable_recording.trial_indices():
                extractor = get_feature_extractor(syllable_recording)
                feats = extractor.extract_features(ecog[start:int(stop + (0.04 * syllable_recording.fs)), :])
                data.append(feats)

            normalization_data = np.concatenate(data)
            self[day] = (np.mean(normalization_data, axis=0), np.std(normalization_data, axis=0))


def main(out_base_path: Path, norm_dir: Path, folders: List[Path]):
    normalization_recordings = norm_dir.glob("**/*.mat")

    syllable_repetitions = {path.parent.name: path for path in normalization_recordings}
    z_score_mapping = ZScoresFromSyllableRepetitions(syllable_recordings=syllable_repetitions, show_pbar=True)

    accumulative_audio_sum = 0.0
    for folder in folders:
        mat_files = list(folder.glob("**/*.mat"))
        wav_files = [mat_file.with_suffix(".wav") for mat_file in mat_files]

        for mat_file, wav_file in zip(mat_files, wav_files):
            if mat_file.parent.name not in z_score_mapping:
                logger.warning(f"No normalization data for {mat_file.parent.name}. Skipping it!")
                continue

            pipeline = FeatureExtractionPipeline(mat_filename=mat_file, wav_filename=wav_file, min_trial_length=2.5)
            ecog = pipeline.get_features(show_pbar=True)
            targ = pipeline.get_lpc_coefficients(show_pbar=True)
            nvad = pipeline.get_vad_labels(show_pbar=True)
            tids = pipeline.get_trial_ids()
            accumulative_audio_sum += pipeline.accumulative_audio_duration()

            # Normalization for ecog data
            norm_means, norm_stds = z_score_mapping[mat_file.parent.name]
            ecog = (ecog - norm_means) / norm_stds

            # Store parameters in HDF container
            out_filename = Path(os.path.join(out_base_path.as_posix(), mat_file.parent.name,
                                             mat_file.with_suffix('.hdf').name))
            os.makedirs(out_filename.parent, exist_ok=True)
            parameters = dict(hga_activity=ecog, lpc_coefficients=targ, vad_labels=nvad, trial_ids=tids)
            save_data_to_hdf(out_filename.as_posix(), parameters=parameters, overwrite=True)

    logger.info(f"Finished. Total of {accumulative_audio_sum / 60 / 60:.02f}h of speech data.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare the data corpus of the speech data into features and labels for training the neural "
                    "network architectures.")
    parser.add_argument("out_dir", help="Path to the parent directory in which the feature/label HDF files will be "
                                        "stored.")
    parser.add_argument("norm_dir", help="Path to parent directory in which the recording mat files from BCI2000 are "
                                         "stored that will be used to compute the normalization statistics.")
    parser.add_argument("folders", nargs='+', help="List of folders containing the recording mat files from BCI2000.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    logger.info(f'python prepare_corpus.py {args.out_dir} {args.norm_dir} {args.folders}')
    folders = [Path(folder) for folder in args.folders]
    main(out_base_path=Path(args.out_dir), norm_dir=Path(args.norm_dir), folders=folders)
