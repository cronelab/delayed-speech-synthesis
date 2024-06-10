import os
import logging
import argparse
import numpy as np
from scipy.io import loadmat, savemat
from pathlib import Path
from collections import defaultdict
from local.common import EnergyBasedVad
from local.common import SelectElectrodesFromBothGrids
from local.common import SelectElectrodesOverSpeechAreas

logger = logging.getLogger("aggregate_per_day.py")


def aggregate_data(speech_corpus_root: Path, agg_path: Path, timing_path: Path):
    """
    Does the first step for doing the contamination report.
    """
    feature_files = list(speech_corpus_root.rglob('KeywordReading_Overt_R*.mat'))
    groups_by_day = defaultdict(list)
    for feature_file in feature_files:
        day = feature_file.parent.name
        groups_by_day[day].append(feature_file)

    selection_1 = SelectElectrodesFromBothGrids()
    selection_2 = SelectElectrodesOverSpeechAreas()
    for day in groups_by_day.keys():
        brain = []
        audio = []
        sampling_rates = set()
        for keyword_recording in groups_by_day[day]:
            mat = loadmat(keyword_recording.as_posix(), simplify_cells=True)
            fs = mat["parameters"]["SamplingRate"]["NumericValue"]
            gain = selection_1(mat["parameters"]["SourceChGain"]["NumericValue"].reshape((1, -1)))

            ecog = selection_1(mat["signal"]) * gain
            if day in ["2022_10_05", "2022_10_06", "2022_10_10"]:
                contaminated_channel_indices = [46, ]
                selection_3 = np.array(
                    [x for x in selection_2.speech_grid_mapping if x not in np.array(contaminated_channel_indices) - 1])

                ecog = ecog[:, selection_3]

            else:
                ecog = selection_2(ecog)

            brain.append(ecog)
            audio.append(mat["signal"][:, 128] * mat["parameters"]["SourceChGain"]["NumericValue"][128])
            sampling_rates.add(fs)

        if len(sampling_rates) != 1:
            logger.warning("WARNING: Found more than one sampling rate for that particular day!!")

        brain = np.concatenate(brain)
        audio = np.concatenate(audio)

        fs = sampling_rates.pop()
        vad = EnergyBasedVad()
        vad_labels = vad.from_wav(audio, sampling_rate=fs)
        diff = np.where(vad_labels[:-1] != vad_labels[1:])[0] + 1
        diff = diff.astype(np.float32)
        diff[1::2] -= 1
        diff *= 0.01
        timings = np.zeros(shape=(len(diff) // 2, 2), dtype=np.float32)
        timings[:, 0] = diff[0::2]
        timings[:, 1] = diff[1::2]

        mdict = dict(fs=fs, ecog=brain, audio=audio)
        fname = os.path.join(agg_path.as_posix(), f"{day}_KeywordReading_Overt.mat")
        savemat(fname, mdict, format='5', long_field_names=False)

        mdict = dict(timings=timings)
        fname = os.path.join(timing_path.as_posix(), f"{day}_KeywordReading_Overt_timings.mat")
        savemat(fname, mdict, format='5', long_field_names=False)


if __name__ == "__main__":
    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    # read command line arguments
    parser = argparse.ArgumentParser("Prepare the experiment recording files for the matlab package provided by Roussel"
                                     " et al. to conduct acoustic contamination analyses.")
    parser.add_argument("--corpus-root", required=True, help="Path to root directory where the experiment "
                                                             "recording files are stored.")
    parser.add_argument("--acc-path", required=True, help="Output path on where to store the accumulated day-specific "
                                                          "containers.")
    parser.add_argument("--timing-path", required=True, help="Output path on where to store the timing data.")
    args = parser.parse_args()

    speech_corpus_root = Path(args.corpus_root)
    agg_path = Path(args.acc_path)
    timing_path = Path(args.timing_path)

    os.makedirs(agg_path.as_posix(), exist_ok=True)
    os.makedirs(timing_path.as_posix(), exist_ok=True)
    aggregate_data(speech_corpus_root, agg_path, timing_path)
