import argparse
import numpy as np
import math
import tqdm
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import spectrogram
from local.common import ExperimentMapping, SelectElectrodesFromBothGrids, EnergyBasedVad
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Optional


logger = logging.getLogger("power_spectral_analysis.py")


def gen_power_spectral_analysis_figure(mat_file: Path, cal_file: Path, keyword: Optional[str] = None,
                                       out_dir: Optional[Path] = None):
    cal = loadmat(cal_file.as_posix(), simplify_cells=True)
    mat = loadmat(mat_file.as_posix(), simplify_cells=True)

    # Extract channels specific signals
    ch_selection = SelectElectrodesFromBothGrids()
    cal_signals = ch_selection(cal["signal"] * cal["parameters"]["SourceChGain"]["NumericValue"])
    cal_audio = cal["signal"][:, 128] * cal["parameters"]["SourceChGain"]["NumericValue"][128]
    mat_signals = ch_selection(mat["signal"] * mat["parameters"]["SourceChGain"]["NumericValue"])
    mat_audio = mat["signal"][:, 128] * mat["parameters"]["SourceChGain"]["NumericValue"][128]

    # Extract sampling rates
    cal_fs = cal["parameters"]["SamplingRate"]["NumericValue"]
    mat_fs = mat["parameters"]["SamplingRate"]["NumericValue"]

    # Extract trial indices
    cal_experiment_class = ExperimentMapping.get_experiment_class(mat_filename=cal_file)
    cal_experiment = cal_experiment_class(cal['states']['StimulusCode'], ExperimentMapping.extract_stimuli_values(cal))

    mat_experiment_class = ExperimentMapping.get_experiment_class(mat_filename=mat_file)
    mat_experiment = mat_experiment_class(mat['states']['StimulusCode'], ExperimentMapping.extract_stimuli_values(mat))

    # Spectrogram configuration
    window_size = 0.05
    nb_fft_bins = 100
    pre_onset = 0.5
    post_onset = 1.5

    # Compute normalization statistics
    normalization_statistics = np.zeros((len(ch_selection), int(nb_fft_bins // 2) + 1), dtype=np.float32)
    for channel in tqdm.tqdm(range(len(normalization_statistics)), desc="Normalization channels"):
        baseline_spectrograms = []
        for _, start, stop in cal_experiment.get_trial_indices():
            channel_segment = cal_signals[start:stop, channel]
            audio_segment = cal_audio[start:int(stop + 0.04 * cal_fs)]

            _, _, Sxx = spectrogram(channel_segment, fs=cal_fs, window="hann", nfft=nb_fft_bins,
                                    nperseg=int(window_size * cal_fs),
                                    noverlap=int(window_size * cal_fs - 0.01 * cal_fs))
            baseline_spectrograms.append(Sxx)

        normalization_statistics[channel] = np.mean(np.concatenate(baseline_spectrograms, axis=1), axis=-1)

    # Compute normalized spectrogram for each channel (+ 5 to include overlap (0.04 s) frames)
    num_windows = math.floor((pre_onset * mat_fs + post_onset * mat_fs - (window_size * mat_fs)) / (0.01 * mat_fs)) + 5
    nb_pre_onset_frames = math.floor((pre_onset * mat_fs - (window_size * mat_fs)) / (0.01 * mat_fs)) + 5
    nb_post_onset_frames = math.floor((post_onset * mat_fs - (window_size * mat_fs)) / (0.01 * mat_fs)) + 5

    channel_spectrograms = np.zeros((len(ch_selection), int(nb_fft_bins // 2) + 1, num_windows), dtype=np.float32)
    for channel in tqdm.tqdm(range(len(channel_spectrograms)), desc="Channel spectrogram"):
        trial_spectrograms = []

        for label, start, stop in mat_experiment.get_trial_indices():
            if keyword is not None and label != keyword:
                continue

            channel_segment = mat_signals[start:int(stop + post_onset * mat_fs), channel]
            audio_segment = mat_audio[start:int(stop + post_onset * mat_fs)]

            vad = EnergyBasedVad()
            vad_labels = vad.from_wav(audio_segment, sampling_rate=mat_fs).astype(bool)

            _, _, Sxx = spectrogram(channel_segment, fs=mat_fs, window="hann", nfft=nb_fft_bins,
                                    nperseg=int(window_size * mat_fs),
                                    noverlap=int(window_size * mat_fs - 0.01 * mat_fs))

            speech_onset = np.argmax(vad_labels)
            spec = Sxx[:, speech_onset - nb_pre_onset_frames:speech_onset + nb_post_onset_frames]
            trial_spectrograms.append(spec)

        channel_spectrogram = np.mean(np.stack(trial_spectrograms), axis=0)
        baseline = np.tile(normalization_statistics[channel], (channel_spectrogram.shape[1], 1)).T
        normalized_spectrogram = 10 * np.log10(channel_spectrogram / baseline)
        channel_spectrograms[channel, :, :] = normalized_spectrogram

    print(channel_spectrograms.shape)

    # Layout grid for plotting
    upper_grid = np.arange(64) + 64
    upper_grid = upper_grid.reshape((8, 8))
    upper_grid = np.flip(np.flip(upper_grid), axis=1)

    lower_grid = np.arange(64)
    lower_grid = lower_grid.reshape((8, 8))
    lower_grid = np.flip(np.flip(lower_grid), axis=1)

    nan_line = np.ones((1, 8)) * np.NAN
    plot_layout = np.concatenate([upper_grid, nan_line, lower_grid])

    height_ratios = np.ones(17)
    height_ratios[8] = 0.25
    fig, axes = plt.subplots(nrows=17, ncols=8, gridspec_kw={"height_ratios": height_ratios}, figsize=(8.5, 10))
    for i, (row, col) in enumerate(np.ndindex(plot_layout.shape)):
        ax = axes[row, col]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        channel_index = plot_layout[row, col]
        if np.isnan(channel_index):
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            continue

        channel_index = int(channel_index)
        if channel_index in np.array([19, 38, 48, 52]) - 1:
            ax.set_facecolor((1.0, 0.0, 0.0))
            ax.text(.5, .5, "Bad", ha='center', va='center', color="white", fontweight="bold", fontsize=12)
            continue

        im = ax.imshow(channel_spectrograms[channel_index, :, :], aspect="auto", origin="lower", cmap="PiYG",
                       vmin=-4, vmax=4)
        ax.axvline(nb_pre_onset_frames, c="black", linestyle="--", linewidth=1)
        ax.text(.02, .96, f"{channel_index + 1:02d}", ha='left', va='top', transform=ax.transAxes, fontsize=8)

        if channel_index in np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                      17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                      34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50,
                                      51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 68]) - 1:
            pass

        if channel_index in np.array(
                [1, 2, 3, 0, 4, 11, 5, 6, 7, 10, 12, 9, 19, 8, 15, 20, 13, 14, 17, 22, 18, 21, 29, 16,
                 23, 28, 35, 36, 27, 25, 26, 55, 45, 46, 44, 24, 37, 40, 33, 34, 32, 51, 47, 39, 31,
                 54, 53, 30, 48, 38, 43, 41, 52, 61, 59, 62, 49, 66, 60, 63, 58, 50, 42, 56, 67, 57,
                 81, 68]):
            ax.spines["top"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines["right"].set_linewidth(1.5)

            ax.spines["top"].set_color("dodgerblue")
            ax.spines["bottom"].set_color("dodgerblue")
            ax.spines['left'].set_color("dodgerblue")
            ax.spines["right"].set_color("dodgerblue")

        if channel_index == 7:
            ax.set_xticks([0, nb_pre_onset_frames, nb_pre_onset_frames + nb_post_onset_frames])
            ax.set_yticks([0, channel_spectrograms.shape[1]])
            ax.set_xticklabels([-pre_onset, 0, post_onset])
            ax.set_yticklabels([0, 500])
            ax.yaxis.tick_right()
            ax.set_ylabel("Freq. [Hz]", labelpad=5)
            ax.yaxis.set_label_position("right")

        if channel_index == 127:
            axins = inset_axes(ax, width="100%", height="100%", loc='upper left',
                               bbox_to_anchor=(1.15, -7, .25, 8), bbox_transform=ax.transAxes)

            axins.set_xticks([])
            axins.set_yticks([])
            fig.colorbar(im, cax=axins, orientation="vertical")
            axins.set_ylabel("Change from non-speech baseline [dB]")

        if channel_index == 3:
            ax.set_xlabel("Time [s]", labelpad=5)
            ax.xaxis.set_label_coords(1.1, -0.25)

    plt.subplots_adjust(top=0.96, bottom=0.04, left=0.03, right=0.87, wspace=0.06, hspace=0.09)
    if out_dir:
        plt.savefig(out_dir / "supplementary_fig_2.png", dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    # read command line arguments
    parser = argparse.ArgumentParser("Generate the power spectral analysis figure from the supplementary material.")
    parser.add_argument("mat", help="Path to a .mat file from the KeywordReading task.")
    parser.add_argument("cal", help="Path to a .mat file from the SyllableRepetition task from the same "
                                    "day as the KeywordReading task.")
    parser.add_argument("--out", "-o", help="Output folder path.")
    args = parser.parse_args()

    cal_file = Path(args.cal)
    mat_file = Path(args.mat)

    gen_power_spectral_analysis_figure(mat_file, cal_file, out_dir=Path(args.out) if args.out else None)
