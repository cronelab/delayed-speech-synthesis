import zmq
import struct
import time
import os
import mne
import sys
import LPCNet
import torch
import torch.nn as nn
import logging
import numpy as np
import zmq.asyncio
import ezmsg.core as ez
from typing import AsyncGenerator, Optional, List, Callable, Iterable
from ezmsg.eeg.eegmessage import TimeSeriesMessage
from dataclasses import replace
from io import TextIOWrapper
from pathlib import Path
from scipy.io.wavfile import write as wavwrite
from scipy.signal import sosfilt, sosfilt_zi
from local.common import VoiceActivityDetectionSmoothing, SpeechSegmentHistory
from functools import reduce
from hga_optimized import compute_log_power_features, WarmStartFrameBuffer


logger = logging.getLogger("units.py")


class ClosedLoopMessage(TimeSeriesMessage):
    """
    Extension to the TimeseriesMessage from the ezmsg.eeg module to contain an optional received_at variable which can
    be used to propagate the timestamp of previous messages to compute the processing time at a final unit.
    """
    received_at: Optional[float] = None
    previous_frames: Optional[float] = None


# region BCI2000 -> ZMQ connector
class ZMQConnectorSettings(ez.Settings):
    fs: int
    port: int = 5556
    address: str = 'localhost'


class ZMQConnectorState(ez.State):
    context: Optional[zmq.asyncio.Context] = None
    socket: Optional[zmq.asyncio.Socket] = None
    header: struct.Struct = struct.Struct('=BBB HH')
    topic: Optional[bytes] = None


class ZMQConnector(ez.Unit):
    """
    Temporary connector class for a better understanding of what is going on behind th scenes. Might be removed later.
    """
    SETTINGS: ZMQConnectorSettings
    STATE: ZMQConnectorState

    OUTPUT = ez.OutputStream(ClosedLoopMessage)

    def initialize(self) -> None:
        # Packet decoding
        self.STATE.topic = struct.Struct('=BBB').pack(4, 1, 2)

        # ZMQ networking
        address = f'tcp://{self.SETTINGS.address}:{self.SETTINGS.port}'
        self.STATE.context = zmq.asyncio.Context()
        self.STATE.socket = self.STATE.context.socket(zmq.SUB)
        self.STATE.socket.setsockopt(zmq.RCVHWM, 1)
        self.STATE.socket.connect(address)
        self.STATE.socket.subscribe(self.STATE.topic)

    def shutdown(self) -> None:
        self.STATE.socket.unsubscribe(self.STATE.topic)
        self.STATE.socket.close()
        self.STATE.context.destroy()

    def interpret_bytes(self, data: bytes) -> np.ndarray:
        descriptor, supplement, dtype, n_channels, n_samples = self.STATE.header.unpack(data[:self.STATE.header.size])
        array = np.frombuffer(data[self.STATE.header.size:], dtype=np.float32).reshape(n_channels, n_samples)
        array = np.transpose(array).astype(np.float64, order='C', copy=True)
        return array

    @ez.publisher(OUTPUT)
    async def process(self) -> AsyncGenerator:
        while not self.STATE.socket.closed:
            data = await self.STATE.socket.recv()
            data = self.interpret_bytes(data)
            yield self.OUTPUT, ClosedLoopMessage(data=data, fs=self.SETTINGS.fs, received_at=time.time())
# endregion


# region Feature extraction
Transforms = Optional[List[Callable]]


class HighGammaExtractor:
    """
    Base class for the feature extraction that provides common functionalities for both the online and the offline
    computation.
    """
    def __init__(self, fs, nb_electrodes, window_length=0.05, window_shift=0.01, l_freq: int = 70, h_freq: int = 170,
                 pre_transforms: Optional[List[Callable]] = None, post_transforms: Optional[List[Callable]] = None):
        self.fs = fs
        self.nb_electrodes = nb_electrodes
        self.window_length = window_length
        self.window_shift = window_shift
        self.model_order = 4
        self.step_size = 5
        self.pre_transform = pre_transforms
        self.post_transform = post_transforms
        self.framebuffer = WarmStartFrameBuffer(frame_length=window_length, frame_shift=window_shift, fs=fs,
                                                nb_channels=nb_electrodes)

        if self.pre_transform is not None:
            self.pre_transform = self._compose_functions(*pre_transforms)
        if self.post_transform is not None:
            self.post_transform = self._compose_functions(*post_transforms)

        if not ((60 < l_freq < 120) or (120 < h_freq < 180)):
            logger.warning('l_freq and h_freq seem not to be in the recommended ranges!!')

        # Initialize filters and filter states
        iir_params = {'order': 8, 'ftype': 'butter'}
        self.hg_filter = self.create_filter(fs, l_freq, h_freq, method='iir', iir_params=iir_params)["sos"]
        self.fh_filter = self.create_filter(fs, 122, 118, method='iir', iir_params=iir_params)["sos"]

        hg_state = sosfilt_zi(self.hg_filter)
        fh_state = sosfilt_zi(self.fh_filter)

        self.hg_state = np.repeat(hg_state, nb_electrodes, axis=-1).reshape([hg_state.shape[0], hg_state.shape[1], -1])
        self.fh_state = np.repeat(fh_state, nb_electrodes, axis=-1).reshape([fh_state.shape[0], fh_state.shape[1], -1])

    @staticmethod
    def _compose_functions(*functions):
        return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

    @staticmethod
    def create_filter(sr, l_freq, h_freq, method='fir', iir_params=None):
        iir_params, method = mne.filter._check_method(method, iir_params)
        filt = mne.filter.create_filter(None, sr, l_freq, h_freq, 'auto', 'auto',
                                        'auto', method, iir_params, 'zero', 'hamming', 'firwin', verbose=30)
        return filt

    def extract_features(self, data: np.ndarray):
        # Apply pre-transforms
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Extract high-gamma activity
        data, self.hg_state = sosfilt(self.hg_filter, data, axis=0, zi=self.hg_state)
        data, self.fh_state = sosfilt(self.fh_filter, data, axis=0, zi=self.fh_state)

        # Compute power features
        data = self.framebuffer.insert(data)
        data = compute_log_power_features(data, self.fs, self.window_length, self.window_shift)

        # Apply post-transform
        if self.post_transform is not None:
            data = self.post_transform(data)
        return data


class HighGammaActivitySettings(ez.Settings):
    """
    Settings for the high-gamma activity unit. Window length and window shift need to be provided in seconds.
    """
    fs: int
    nb_electrodes: int
    window_length: float = 0.05
    window_shift: float = 0.01
    l_freq: int = 70
    h_freq: int = 170
    pre_transforms: Transforms = None
    post_transforms: Transforms = None


class HighGammaActivityState(ez.State):
    """
    Filter states of the high-gamma filter and the one for the first harmonic.
    """
    hg_extractor: Optional[HighGammaExtractor] = None


class HighGammaActivity(ez.Unit):
    """
    Unit for extraction of the high-gamma band in the range of 70 to 170 Hz. It also filters out the first harmonic
    of the line noise and provides options on how to extract features and how to transform them.
    """
    SETTINGS: HighGammaActivitySettings
    STATE: HighGammaActivityState

    INPUT: ez.InputStream = ez.InputStream(TimeSeriesMessage)
    OUTPUT: ez.OutputStream = ez.OutputStream(TimeSeriesMessage)

    def initialize(self) -> None:
        self.STATE.hg_extractor = HighGammaExtractor(
            fs=self.SETTINGS.fs, nb_electrodes=self.SETTINGS.nb_electrodes,
            window_length=self.SETTINGS.window_length, window_shift=self.SETTINGS.window_shift,
            pre_transforms=self.SETTINGS.pre_transforms, post_transforms=self.SETTINGS.post_transforms
        )

    @ez.publisher(OUTPUT)
    @ez.subscriber(INPUT)
    async def process(self, msg: TimeSeriesMessage) -> AsyncGenerator:
        features = self.STATE.hg_extractor.extract_features(msg.data)
        yield self.OUTPUT, replace(msg, data=features, fs=1/self.SETTINGS.window_shift)
# endregion


# region Logging units
class LoggerSettings(ez.Settings):
    """
    General settings for speech related message loggers
    """
    filename: str
    overwrite: bool
    config_filename: Optional[str] = None


class BinaryLoggerState(ez.State):
    """
    The binary logger state only contains the file descriptor which updates its position on each write operation
    """
    file_descriptor: Optional[TextIOWrapper] = None
    shape: Optional[Iterable[int]] = None


class BinaryLogger(ez.Unit):
    """
    Write the data field from a TimeSeriesMessage into a binary log file. For restoring the file contents, one dimension
    (for example the number of columns) and the data type need to be known. With this information, np.frombuffer() or
    np.fromfile can be used to restore all written data into a single array.

    Example:
        data = np.fromfile(log_filename, dtype=...).reshape((-1, ...)).astype(dtype, order='C', copy=True)
    """
    SETTINGS: LoggerSettings
    STATE: BinaryLoggerState
    INPUT: ez.InputStream = ez.InputStream(TimeSeriesMessage)

    def initialize(self) -> None:
        """
        Check if filename is a valid path (otherwise create the path) and check if the specified filename is already
        present. If that is the case and overwrite iin settings is set to False, an exception is raised.
        """
        filename = os.path.abspath(self.SETTINGS.filename)  # TODO use pathlib
        extension = os.path.basename(filename).split('.')[-1]
        storage_dir = os.path.dirname(filename)

        if not os.path.exists(storage_dir): os.makedirs(storage_dir)
        if os.path.exists(filename) and os.path.isfile(filename) and not self.SETTINGS.overwrite:
            raise PermissionError(f'The specified .{extension} file already exists and overwrite is disabled.')

        self.STATE.file_descriptor = open(filename, mode='wb')

    def shutdown(self) -> None:
        """
        Write the remaining data in the buffer to file and close the descriptor.
        """
        self.STATE.file_descriptor.flush()
        self.STATE.file_descriptor.close()

    @ez.subscriber(INPUT)
    async def write(self, message: TimeSeriesMessage) -> None:
        if self.STATE.shape is None:
            self.STATE.shape = list(message.data.shape)
            if len(self.STATE.shape) > 1:
                self.STATE.shape.pop(message.time_dim)
        self.STATE.file_descriptor.write(message.data.tobytes())


class VoiceActivityDetectionLoggerState(ez.State):
    """
    Te VAD logger state contains the file descriptor which is used to write vad entries in the .lab file format
    (tab separated textfile containing the columns start, stop and label, see:
    https://manual.audacityteam.org/man/importing_and_exporting_labels.html for a detailed description).
    """
    file_descriptor: Optional[TextIOWrapper] = None


class VoiceActivityDetectionLogger(ez.Unit):
    """
    Logger unit which directly output acoustic samples into awave file format.
    """
    SETTINGS: LoggerSettings
    STATE: VoiceActivityDetectionLoggerState
    INPUT: ez.InputStream = ez.InputStream(ClosedLoopMessage)

    def initialize(self) -> None:
        """
        Setup up the file descriptor for logging the voice activity segments.
        """
        filename = os.path.abspath(self.SETTINGS.filename)
        storage_dir = os.path.dirname(filename)

        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        if os.path.exists(filename) and os.path.isfile(filename) and not self.SETTINGS.overwrite:
            raise PermissionError('The specified .lab file already exists and overwrite is disabled.')

        self.STATE.file_descriptor = open(filename, mode='w')

    def shutdown(self) -> None:
        """
        Make sure to write all remaining entries to file before closing.
        """
        self.STATE.file_descriptor.flush()
        self.STATE.file_descriptor.close()

    @ez.subscriber(INPUT)
    async def write(self, message: ClosedLoopMessage) -> None:
        """
        Message contains all the high-gamma activity frames that have been extracted from the voice activity
        detection module.
        """
        start = message.previous_frames * 0.01
        stop = (message.previous_frames + len(message.data)) * 0.01
        self.STATE.file_descriptor.write(f"{start:.02f}\t{stop:.02f}\t{len(message.data)} frames\n")


class DelayedWavLoggerSettings(ez.Settings):
    """
    Settings class for the Delayed wav logger for saving each speech segment in ints separate file.
    """
    base_path: Path
    overwrite: bool
    prefix: Optional[str] = None


class DelayedWavLoggerState(ez.State):
    """
    The state only contains a counter for the number of saved words to make comparison with original possible.
    """
    speech_segment_counter: int = 1


class DelayedWavLogger(ez.Unit):
    """
    The delayed wav logger takes a synthesized speech segment and saves the whole segment to file using a file pattern
    and that includes the number of the synthesized word.
    """
    SETTINGS: DelayedWavLoggerSettings
    STATE: DelayedWavLoggerState
    INPUT: ez.InputStream = ez.InputStream(TimeSeriesMessage)

    def initialize(self) -> None:
        """
        In case the base_path does not exist, create it.
        """
        os.makedirs(self.SETTINGS.base_path, exist_ok=True)

    @ez.subscriber(INPUT)
    async def write(self, message: TimeSeriesMessage) -> None:
        """
        Message contains the speech signal to be saved as a wav file.
        """
        prefix = self.SETTINGS.prefix if self.SETTINGS.prefix is not None else ""
        filename = os.path.join(self.SETTINGS.base_path.as_posix(),
                                f"{prefix}_{self.STATE.speech_segment_counter:05d}.wav")

        self.STATE.speech_segment_counter += 1
        if not (os.path.isfile(filename) and not self.SETTINGS.overwrite):
            wavwrite(filename, 16000, message.data)
# endregion


# region Neural VAD and speech decoding units
class FilterSpeechSegmentsSettings(ez.Settings):
    """
    Settings class for filtering consecutive speech segments.
    """
    nb_features: int
    fs: int
    vad_architecture: nn.Module
    buffer_size: int
    context_frames: int = 0
    vad_weights_path: Optional[Path] = None
    vad_parameters: Optional[dict] = None


class FilterSpeechSegmentsState(ez.State):
    """
    State class for filtering consecutive speech segments contains a ringbuffer that tracts past speech frames and a
    model that classifies high-gamma frames as speech/non-speech. A smoothing component counters fluctuations in the
    classification model to be identified as separate speech segments.
    """
    device: str
    history: SpeechSegmentHistory
    smoothing: VoiceActivityDetectionSmoothing
    vad_model: nn.Module
    vad_state: torch.Tensor
    frame_counter: int


class FilterSpeechSegments(ez.Unit):
    """
    Filter unit that extract consecutive segments of speech in the neural data and outputs only those identified.
    """
    SETTINGS: FilterSpeechSegmentsSettings
    STATE: FilterSpeechSegmentsState

    INPUT: ez.InputStream = ez.InputStream(ez.Message)
    OUTPUT: ez.OutputStream = ez.OutputStream(ez.Message)

    def initialize(self) -> None:
        # Determine device
        self.STATE.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Set up the history ringbuffer
        self.STATE.history = SpeechSegmentHistory(nb_features=self.SETTINGS.nb_features,
                                                  buffer_size=self.SETTINGS.buffer_size,
                                                  context=self.SETTINGS.context_frames)

        # Set up the smoothing ringbuffer
        self.STATE.smoothing = VoiceActivityDetectionSmoothing(nb_features=self.SETTINGS.nb_features, context_frames=5)

        # Initialize the VAD model
        params = self.SETTINGS.vad_parameters if self.SETTINGS.vad_parameters is not None else dict()
        self.STATE.vad_model = self.SETTINGS.vad_architecture(**params).to(self.STATE.device)
        if self.SETTINGS.vad_weights_path is not None:
            self.STATE.vad_model.load_state_dict(
                torch.load(self.SETTINGS.vad_weights_path.as_posix(), map_location=self.STATE.device)
            )

        self.STATE.vad_state = self.STATE.vad_model.create_new_initial_state(batch_size=1, device=self.STATE.device)
        self.STATE.vad_model.eval()
        self.STATE.frame_counter = 0

    @ez.publisher(OUTPUT)
    @ez.subscriber(INPUT)
    async def process(self, msg: ClosedLoopMessage) -> AsyncGenerator:
        ecog_tensor = torch.from_numpy(np.expand_dims(msg.data, 0)).float().to(self.STATE.device)  # Batch x Time x Feat
        predictions, self.STATE.vad_state = self.STATE.vad_model(ecog_tensor, self.STATE.vad_state)
        predictions = torch.argmax(predictions, dim=2).flatten().detach().cpu().numpy()

        # Smooth labels to avoid break of speech segments from misclassifications
        data, predictions = self.STATE.smoothing.insert(data=msg.data, speech_labels=predictions)

        # Insert frames into ringbuffer
        speech_segments = self.STATE.history.insert(data=data, speech_labels=predictions)

        # If one or more speech segments have been completed, output them for the next unit
        self.STATE.frame_counter += len(msg.data)
        for segment in speech_segments:
            previous_frames = self.STATE.frame_counter - len(segment) - (len(msg.data) - np.count_nonzero(predictions))
            yield self.OUTPUT, replace(msg, data=segment, fs=100, previous_frames=previous_frames)


class RecurrentNeuralDecodingModelSettings(ez.Settings):
    """
    Path to model weights reflects the path in which state_dict is stored. The model class variable refers to
    the pytorch module that can instantiate an architecture. If this model requires parameters to be set in its __ini__
    function, these can be set via the params optional dict.
    """
    path_to_model_weights: Optional[str]
    model: nn.Module
    params: Optional[dict]
    config_filename: Optional[str] = None


class RecurrentNeuralDecodingModelState(ez.State):
    """
    The neural decoding model state keeps track of the instantiated pytorch model and the device on which the model
    was deployed on.
    """
    decoding_model: Optional[nn.Module] = None
    device: Optional[str] = None
    H: Optional[torch.Tensor] = None


class RecurrentNeuralDecodingModel(ez.Unit):
    SETTINGS: RecurrentNeuralDecodingModelSettings
    STATE: RecurrentNeuralDecodingModelState

    INPUT = ez.InputStream(TimeSeriesMessage)
    OUTPUT = ez.OutputStream(TimeSeriesMessage)

    def initialize(self) -> None:
        """
        Instantiate pytorch model with specified parameters from the settings dataclass and deploys it on the cuda
        device. If no cuda device is accessible, it gets deployed on the cpu.
        """
        params = self.SETTINGS.params if self.SETTINGS.params is not None else dict()

        # Determine device
        self.STATE.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load pytorch architecture
        self.STATE.decoding_model = self.SETTINGS.model(**params).to(self.STATE.device)
        if self.SETTINGS.path_to_model_weights is not None:
            self.STATE.decoding_model.load_state_dict(torch.load(self.SETTINGS.path_to_model_weights,
                                                                 map_location=self.STATE.device))
        self.STATE.decoding_model.eval()
        self.STATE.H = self.STATE.decoding_model.create_new_initial_state(batch_size=1, device=self.STATE.device)

    @ez.subscriber(INPUT)
    @ez.publisher(OUTPUT)
    async def decode(self, msg: TimeSeriesMessage) -> AsyncGenerator:
        """
        Message contains a consecutive speech segments that will be transformed into LPC features in one step.
        """
        ecog_tensor = torch.from_numpy(np.expand_dims(msg.data, 0)).float().to(self.STATE.device)
        predictions, self.STATE.H = self.STATE.decoding_model(ecog_tensor, self.STATE.H)
        predictions = predictions.detach().cpu().numpy()
        predictions = np.squeeze(predictions, axis=0)
        self.STATE.H = self.STATE.decoding_model.create_new_initial_state(batch_size=1, device=self.STATE.device)
        yield self.OUTPUT, replace(msg, data=predictions, fs=100)
# endregion


# region Vocoder unit
class LPCNetState(ez.State):
    lpcnet: Optional[LPCNet.LPCNet] = None


class DelayedLPCNetVocoder(ez.Unit):
    STATE: LPCNetState

    INPUT: ez.InputStream = ez.InputStream(TimeSeriesMessage)
    OUTPUT: ez.OutputStream = ez.OutputStream(TimeSeriesMessage)

    def initialize(self) -> None:
        self.STATE.lpcnet = LPCNet.LPCNet()

    def shutdown(self) -> None:
        del self.STATE.lpcnet

    @ez.subscriber(INPUT)
    @ez.publisher(OUTPUT)
    async def synthesize(self, msg: TimeSeriesMessage) -> AsyncGenerator:
        message = msg.data.astype(np.float32)
        acoustic_signal = []
        for i in range(len(message)):
            acoustic_signal.append(self.STATE.lpcnet.synthesize(message[i, :]))

        acoustic_signal = np.hstack(acoustic_signal)
        yield self.OUTPUT, replace(msg, data=acoustic_signal, fs=16000)
# endregion


# region Output unit
class DelayedStdoutForSoX(ez.Unit):
    """
    Print speech segment to console to be reinterpreted by SoX
    """
    INPUT: ez.InputStream = ez.InputStream(ClosedLoopMessage)

    @ez.subscriber(INPUT)
    async def print(self, msg: ClosedLoopMessage) -> None:
        sys.stdout.buffer.write(msg.data.tobytes())
        sys.stdout.flush()
# endregion
