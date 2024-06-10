import argparse
import logging
import numpy as np
import struct
import time
import zmq
from scipy.io import loadmat
from typing import List, Tuple, Dict


logger = logging.getLogger('tools:development-amplifier')


class BCI2000Package(struct.Struct):
    """
    Structure of a BCI2000 GenericSignal package to be sent over network protocol.
    """
    def __init__(self, nb_channels, nb_samples):
        package_structure = f'=BBB HH {nb_channels * nb_samples}f'
        self.header_info = (4, 1, 2, nb_channels, nb_samples)
        super().__init__(package_structure)

    def pack(self, payload: np.ndarray) -> bytes:
        package_data = (*self.header_info, *payload.flatten().tolist())
        return super().pack(*package_data)


def extract_stimuli_values(mat) -> List[str]:
    try:
        stimuli = mat['parameters']['Stimuli']['Value']
    except KeyError:
        stimulus_codes = mat['states']['StimulusCode']
        stimuli = np.asarray([f'Unknown stimulus {identifier}' for identifier in np.unique(stimulus_codes) if identifier != 0])

    if stimuli.ndim == 1:
        return [stimuli[0]]
    else:
        return stimuli[0].tolist()


class Amplifier:
    """
    Simulated amplifier which streams neural data on the ZMQ interface.
    """
    def __init__(self, mat_file, package_size, loop=False, seconds=0, port=5556, epsilon=0.0000001):
        self.mat_file = mat_file
        self.package_size = package_size
        self.epsilon = epsilon
        self.loop = loop
        self.seconds = seconds

        # Read data from BCI2000 .mat file
        self.ecog, self.fs, self.stim_codes, self.stimuli = self._read_mat()
        nb_channels = self.ecog.shape[1]
        if seconds > 0:
            self.ecog = self.ecog[:int(seconds * self.fs)]

        # Network configuration
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.bind(f'tcp://127.0.0.1:{self.port}')
        time.sleep(0.5)  # Give some time to bind the socket to the port and not drop the first message(s)

        # Package construction variables
        self.ecog_sample_index = 0
        self.bci2000package = BCI2000Package(nb_channels=nb_channels, nb_samples=self.package_size)

        # Spin-loop variables
        self.package_counter = 0
        self.sample_counter = 0
        self.time_val = None  # time.time()
        self.time_val_initial = None  # self.time_val

        logger.info(f'Initialized simulated amplifier for sending packets of neural data over ZMQ '
                    f'(Packet size: {self.package_size}, Port: {self.port}, Rate: {self.fs}).')

    def __del__(self):
        self.socket.close()
        self.context.destroy()
        uptime = time.time() - self.time_val_initial
        logger.info(f'Ended after {uptime:.01f} seconds [{self.sample_counter} samples sent '
                    f'in a total of {self.package_counter} packets].')

    def _read_mat(self) -> Tuple[np.ndarray, int, np.ndarray, Dict[int, str]]:
        mat = loadmat(self.mat_file, simplify_cells=True)
        fs = mat['parameters']['SamplingRate']['NumericValue']
        ecog = mat['signal']
        gain = mat['parameters']['SourceChGain']['NumericValue']
        stim = extract_stimuli_values(mat)
        stim = {(index+1): item for index, item in enumerate(stim)}
        code = mat['states']['StimulusCode']

        ecog = ecog * gain
        ecog = ecog.astype(np.float32, copy=True, order='C')
        return ecog, fs, code, stim

    def reset(self):
        self.ecog_sample_index = 0
        self.sample_counter = 0
        self.time_val = time.time()
        self.time_val_initial = self.time_val

    def stream(self):
        logger.info(f'Streaming!')

        # Initialize variables for logging stimulus code changes
        diff = np.where(self.stim_codes[:-1] != self.stim_codes[1:])[0] + 1
        stim_code_index = 0

        self.time_val = time.time()
        self.time_val_initial = self.time_val
        while True:
            if not self.loop and self.ecog_sample_index >= len(self.ecog):
                logger.info('Cancelled streaming due to reaching end of ECoG signal dataset.')
                break

            if self.ecog_sample_index <= diff[stim_code_index] < self.ecog_sample_index + self.package_size:
                code = self.stim_codes[diff[stim_code_index]]
                stim = self.stimuli[code] if code in self.stimuli.keys() else ''
                logger.info(f'Stimulus code changed to: {stim}')
                stim_code_index = (stim_code_index + 1) % len(diff)

            package = self.ecog[self.ecog_sample_index:min(self.ecog_sample_index + self.package_size, len(self.ecog))]
            if len(package) < self.package_size:
                nb_missing_samples = self.package_size - len(package)
                pad_data = self.ecog[:nb_missing_samples] if self.loop else \
                    np.zeros((nb_missing_samples, self.ecog.shape[1]), dtype=self.ecog.dtype)
                package = np.vstack([package, pad_data])
                self.ecog_sample_index = nb_missing_samples

            else:
                self.ecog_sample_index += self.package_size

            # Send via ZMQ
            serialized_package = self.bci2000package.pack(package.T)
            self.socket.send(serialized_package)
            self.package_counter += 1

            # Spin-waiting Loop
            while time.time() - self.time_val < self.package_size / self.fs:
                time.sleep(self.epsilon)

            self.sample_counter += len(package)
            self.time_val = self.time_val_initial + self.sample_counter / self.fs


if __name__ == "__main__":
    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    # read command line arguments
    parser = argparse.ArgumentParser("Starts a simulated amplifier which reads neural data from a provided .mat file "
                                     "and streams them via the ZMQ interface to a receiver. This way, closed-loop "
                                     "systems can be tested without being connected to an actual amplifier.")
    parser.add_argument("mat_file", help="Path to the .mat file that stores mirrors .dat BCI2000 files.")
    parser.add_argument("--package_size", "-p", default=20, help="Number of samples per package sent over ZMQ.")
    parser.add_argument("--loop", "-l", help="Restart after all samples have been sent.", action="store_true")
    parser.add_argument("--seconds", "-s", help="Specifies how many seconds should be sent. Defaults to 0, which means "
                                                "all samples should be sent.", default=0)

    args = parser.parse_args()
    mat_file = args.mat_file
    package_size = int(args.package_size)
    logger.info(f'tools:development-amplifier {mat_file} --package_size {package_size} --loop {args.loop} '
                f'--seconds {args.seconds}')

    amplifier = Amplifier(mat_file=mat_file, package_size=package_size, loop=args.loop, seconds=int(args.seconds))
    try:
        amplifier.stream()
    except KeyboardInterrupt:
        logger.info('Amplifier is going to gracefully close...')
    finally:
        del amplifier
