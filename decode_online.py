import argparse
import configparser
import os
import sys
import logging
import json
import ezmsg.core as ez
import numpy as np
from pathlib import Path
from local.units import ZMQConnector, ZMQConnectorSettings
from local.units import HighGammaActivity, HighGammaActivitySettings
from local.units import FilterSpeechSegments, FilterSpeechSegmentsSettings
from local.units import RecurrentNeuralDecodingModel, RecurrentNeuralDecodingModelSettings
from local.units import BinaryLogger, VoiceActivityDetectionLogger, LoggerSettings
from local.units import DelayedWavLogger, DelayedWavLoggerSettings
from local.units import DelayedLPCNetVocoder, DelayedStdoutForSoX
from local.common import ZScoreNormalization, CommonAverageReferencing
from local.common import SelectElectrodesFromBothGrids, SelectElectrodesOverSpeechAreas
from local.models import UnidirectionalVoiceActivityDetector, BidirectionalSpeechSynthesisModel
from typing import Iterable, Optional, Tuple, Any


logger = logging.getLogger('decode_online.py')


# region BCI System
class NeuroprosthesisSettings(ez.Settings):
    """
    Global settings for the speech neuroprosthesis system.
    """
    destination_dir: str
    address: str
    fs: int
    package_size: int
    bad_channels: Optional[Iterable] = None
    decoding_model_weights: Optional[Path] = None
    vad_model_weights: Optional[Path] = None
    normalization_statistics: Optional[Path] = None
    block_size: int = 256


class Neuroprosthesis(ez.System):
    """
    Closed-loop speech neuroprosthesis system which filters for speech segments using a neural VAD model and propagates
    a complete speech segment to the next unit to synthesize it with a bidirectional recurrent neural network model.
    Reconstructed LPC features will be transformed into acoustic speech using a pretrained LPCNet vocoder.
    """
    CONNECTOR = ZMQConnector()
    FEATURE_EXTRACTOR = HighGammaActivity()
    SPEECH_FILTER = FilterSpeechSegments()
    DECODING_MODEL = RecurrentNeuralDecodingModel()
    WAVEFORM_GENERATOR = DelayedLPCNetVocoder()
    LOUDSPEAKER = DelayedStdoutForSoX()

    # Logging units
    RAW_LOGGER = BinaryLogger()
    HGA_LOGGER = BinaryLogger()
    VAD_LOGGER = VoiceActivityDetectionLogger()
    LPC_LOGGER = BinaryLogger()
    WAV_LOGGER = DelayedWavLogger()

    # Delayed neuroprosthesis settings
    SETTINGS: NeuroprosthesisSettings

    def configure_feature_transforms(self) -> Tuple[Any, Any, int]:
        """
        Configure pre- and post-transform lists to extract high-gamma features.
        """
        select_both_grids = SelectElectrodesFromBothGrids()
        pre_transforms = [select_both_grids, ]

        # Apply CAR filter
        speech_grid = np.flip(np.arange(64, dtype=np.int16).reshape((8, 8)) + 1, axis=0)
        motor_grid = np.flip(np.arange(64, dtype=np.int16).reshape((8, 8)) + 65, axis=0)
        layout = np.arange(128) + 1
        car = CommonAverageReferencing(exclude_channels=[19, 38, 48, 52], grids=[speech_grid, motor_grid],
                                       layout=layout)
        pre_transforms.append(car)

        # Select only relevant electrodes
        channel_selection = SelectElectrodesOverSpeechAreas()
        pre_transforms.append(channel_selection)

        if self.SETTINGS.normalization_statistics is None:
            logger.info("Found no normalization data. Going to use zero-mean and unit-variance.")
            channel_means = np.zeros(len(channel_selection), dtype=np.float32)
            channel_stds = np.ones(len(channel_selection), dtype=np.float32)
        else:
            logger.info(f"Found normalizations statistics in {self.SETTINGS.normalization_statistics.as_posix()}.")
            statistics = np.load(self.SETTINGS.normalization_statistics.as_posix())
            channel_means = statistics[0, :]
            channel_stds = statistics[1, :]

        post_transforms = ZScoreNormalization(channel_means=channel_selection(channel_means.reshape((1, -1))),
                                              channel_stds=channel_selection(channel_stds.reshape((1, -1))))

        return pre_transforms, post_transforms, len(channel_selection)

    def configure(self) -> None:
        # Configure the ZMQ connector for communication with the amplifier
        self.CONNECTOR.apply_settings(ZMQConnectorSettings(
            fs=self.SETTINGS.fs, address=self.SETTINGS.address, port=5556
        ))

        # Settings for extracting high-gamma activity
        pre_transforms, post_transforms, nb_features = self.configure_feature_transforms()
        self.FEATURE_EXTRACTOR.apply_settings(HighGammaActivitySettings(
            fs=self.SETTINGS.fs, nb_electrodes=nb_features,
            pre_transforms=pre_transforms, post_transforms=[post_transforms]
        ))

        # Initialize speech filtering unit
        logger.info(f"VAD model weights: {self.SETTINGS.vad_model_weights}")
        nb_electrodes = len(SelectElectrodesOverSpeechAreas())
        self.SPEECH_FILTER.apply_settings(FilterSpeechSegmentsSettings(
            nb_features=nb_features, fs=self.SETTINGS.fs, buffer_size=2000, context_frames=50,
            vad_architecture=UnidirectionalVoiceActivityDetector,
            vad_weights_path=Path(self.SETTINGS.vad_model_weights),
            vad_parameters=dict(nb_layer=2, nb_hidden_units=150, nb_electrodes=nb_electrodes)
        ))

        # Initialize the decoding model
        logger.info(f"Decoding model weights: {self.SETTINGS.decoding_model_weights}")
        self.DECODING_MODEL.apply_settings(RecurrentNeuralDecodingModelSettings(
            path_to_model_weights=self.SETTINGS.decoding_model_weights, model=BidirectionalSpeechSynthesisModel,
            params=dict(nb_layer=2, nb_hidden_units=100, nb_electrodes=nb_electrodes),
        ))

        # Configure logging units
        # RAW logger: samples x 64
        # HGA logger: samples x 64
        # VAD logger: one detected speech segment per line (start, stop, number of frames)
        # LPC logger: samples x 20
        raw_logger_filename = os.path.join(self.SETTINGS.destination_dir, 'log.raw.f64')
        hga_logger_filename = os.path.join(self.SETTINGS.destination_dir, 'log.hga.f64')
        vad_logger_filename = os.path.join(self.SETTINGS.destination_dir, 'log.vad.lab')
        lpc_logger_filename = os.path.join(self.SETTINGS.destination_dir, 'log.lpc.f32')

        self.RAW_LOGGER.apply_settings(LoggerSettings(filename=raw_logger_filename, overwrite=True))  # Raw ECoG signals
        self.HGA_LOGGER.apply_settings(LoggerSettings(filename=hga_logger_filename, overwrite=True))  # High-y features
        self.VAD_LOGGER.apply_settings(LoggerSettings(filename=vad_logger_filename, overwrite=True))  # VAD segments
        self.LPC_LOGGER.apply_settings(LoggerSettings(filename=lpc_logger_filename, overwrite=True))  # LPC coefficients

        wav_storage_dir = Path(os.path.join(self.SETTINGS.destination_dir, "reco"))
        self.WAV_LOGGER.apply_settings(DelayedWavLoggerSettings(base_path=wav_storage_dir, prefix="reco",
                                                                overwrite=True))  # Acoustic speech

    # Define Connections
    def network(self) -> ez.NetworkDefinition:
        return (
            # Main route
            (self.CONNECTOR.OUTPUT, self.FEATURE_EXTRACTOR.INPUT),
            (self.FEATURE_EXTRACTOR.OUTPUT, self.SPEECH_FILTER.INPUT),
            (self.SPEECH_FILTER.OUTPUT, self.DECODING_MODEL.INPUT),
            (self.DECODING_MODEL.OUTPUT, self.WAVEFORM_GENERATOR.INPUT),
            (self.WAVEFORM_GENERATOR.OUTPUT, self.LOUDSPEAKER.INPUT),

            # Connect the waveform generation component both with the loudspeaker and write acoustic samples to file
            (self.CONNECTOR.OUTPUT, self.RAW_LOGGER.INPUT),
            (self.FEATURE_EXTRACTOR.OUTPUT, self.HGA_LOGGER.INPUT),
            (self.SPEECH_FILTER.OUTPUT, self.VAD_LOGGER.INPUT),
            (self.DECODING_MODEL.OUTPUT, self.LPC_LOGGER.INPUT),
            (self.WAVEFORM_GENERATOR.OUTPUT, self.WAV_LOGGER.INPUT),
        )
# endregion

def main(settings: NeuroprosthesisSettings) -> None:
    system = Neuroprosthesis(settings)
    ez.run_system(system)


def build_neuroprostetics_settings(settings_filename: str, run_name: str, overwrite: bool) -> NeuroprosthesisSettings:
    settings_config = configparser.ConfigParser()
    settings_config.read(settings_filename)

    # Load path to model weights if provided, otherwise None
    model_weights_path = settings_config.get('Decoding', 'decoding_model_weights')
    decoding_model_weights = None if model_weights_path == "" else Path(model_weights_path)

    # Load path to model weights if provided, otherwise None
    model_weights_path = settings_config.get('Decoding', 'vad_model_weights')
    vad_model_weights = None if model_weights_path == "" else Path(model_weights_path)

    # Load bad channels if present, otherwise None
    bad_channels_entry = settings_config.get('Decoding', 'bad_channels')
    bad_channels = None if bad_channels_entry == "" else json.loads(bad_channels_entry)

    # Load path to normalization statistics if provided, otherwise None
    normalization_statistics_entry = settings_config.get('Decoding', 'initial_normalization_statistics')
    normalization_statistics = None if normalization_statistics_entry == "" else Path(normalization_statistics_entry)

    # Set destination dir
    base_out_dir = settings_config.get('Decoding', 'base_out_dir')
    destination_dir = os.path.join(base_out_dir, run_name)

    settings = NeuroprosthesisSettings(
        destination_dir=destination_dir,
        address=settings_config.get('Decoding', 'address'),
        fs=settings_config.getint('Decoding', 'fs'),
        package_size=settings_config.getint('Decoding', 'package_size'),
        bad_channels=bad_channels,
        decoding_model_weights=decoding_model_weights,
        vad_model_weights=vad_model_weights,
        normalization_statistics=normalization_statistics,
        block_size=settings_config.getint('Decoding', 'block_size'),
    )

    return settings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Real-time speech synthesis from neural signals with delayed acoustic feedback.')
    parser.add_argument('config', help='Path of the config file on how to set up the BCI system.')
    parser.add_argument('--run', required=False, default='test_run',
                        help='Name of the run folder in which (intermediate) results are stored.')
    parser.add_argument('--overwrite', required=False, default=False, action='store_true',
                        help='Specify if that run already exists if it should be overwritten.')

    args = parser.parse_args()
    settings = build_neuroprostetics_settings(args.config, args.run, args.overwrite)
    try:
        os.makedirs(settings.destination_dir, exist_ok=args.overwrite)
    except FileExistsError:
        logger.error('The file path of the destination directory already exists and the --overwrite flag is not set.')
        exit(1)

    # initialize logging handler
    log_filename = os.path.join(settings.destination_dir, 'log.run.txt')
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S',
                        handlers=[logging.FileHandler(log_filename, 'w+'), logging.StreamHandler(sys.stderr)])

    overwrite = "--overwrite" if args.overwrite else ""
    logger.info(f'python decode_online.py {args.config} --run {args.run} {overwrite}')
    logger.info(f"Setting destination dir to {settings.destination_dir}")

    main(settings)
