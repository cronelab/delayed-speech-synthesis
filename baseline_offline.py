import argparse
import configparser
import logging
import os
import sys
import numpy as np
from datetime import datetime
from typing import Tuple
from local.common import BCI2000MatFile
from prepare_corpus import get_feature_extractor


logger = logging.getLogger("baseline_offline.py")


def get_default_session_name() -> str:
    return datetime.now().strftime("%Y_%m_%d")


def get_paths(settings_filename: str) -> Tuple[str, str]:
    settings_config = configparser.ConfigParser()
    settings_config.read(settings_filename)

    # Compile path to session dir
    base_path = settings_config.get("Normalization", "base_path")
    session = get_default_session_name() if settings_config.get("Normalization", "session") == "" \
        else settings_config.get("Normalization", "session")
    session = os.path.join(base_path, session)

    # Get normalization file if provided
    norm_file = None if settings_config.get("Normalization", "normalization_file") == "" \
        else settings_config.get("Normalization", "normalization_file")

    return session, norm_file


def main(session: str, norm_file: str):
    logger.info(f"Processing {norm_file}")
    mat_file = BCI2000MatFile(mat_filename=norm_file)
    ecog = mat_file.signals()

    if mat_file.bad_channels() is not None:
        logger.warning(f"Found the following bad channels in the normalization data: {mat_file.bad_channels()}")

    trials = []
    logger.info("Aggregating trails on which normalization statistics will be computed.")
    for _, start, stop in mat_file.trial_indices():
        extractor = get_feature_extractor(mat_file)
        feats = extractor.extract_features(ecog[start:int(stop + (0.04 * mat_file.fs)), :])
        trials.append(feats)

    logger.info("Compute normalization statistics.")
    normalization_data = np.concatenate(trials)
    mean = np.mean(normalization_data, axis=0)
    std = np.std(normalization_data, axis=0)

    out_filename = os.path.join(session, "normalization.npy")
    logger.info(f"Normalization statistics will be stored in {out_filename}")
    normalization_statistics = np.vstack([mean, std])
    np.save(out_filename, normalization_statistics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute z-score statistics from .mat file")
    parser.add_argument('config', help='Path of the config file.')
    parser.add_argument('--norm', help='Path to the .mat file (overwrites the field normalization_file in config).')
    parser.add_argument('--overwrite', required=False, default=False, action='store_true',
                        help='Specify if the session folder might get overwritten.')
    args = parser.parse_args()

    # Get session dir
    session_dir, norm_file = get_paths(settings_filename=args.config)
    os.makedirs(session_dir, exist_ok=args.overwrite)

    if args.norm is not None:
        norm_file = args.norm

    # Initialize logging handler
    log_filename = os.path.join(session_dir, 'log.normalization.txt')
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S',
                        handlers=[logging.FileHandler(log_filename, 'w+'), logging.StreamHandler(sys.stderr)])

    if norm_file is None:
        logger.error("No normalization file provided. Script will exit.")
        exit(1)

    # Get session dir
    overwrite = " --overwrite" if args.overwrite else ""
    logger.info(f"python baseline_offline.py {args.config} --norm {norm_file}" + overwrite)
    logger.info(f"Session path: {session_dir}.")
    main(session=session_dir, norm_file=norm_file)
