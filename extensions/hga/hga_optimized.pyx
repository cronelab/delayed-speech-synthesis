from libc.math cimport round, floor, pow, log
import cython
import numpy as np
cimport numpy as np
np.import_array()


@cython.boundscheck(False)
cdef double array_sum_and_power(double [:, :] data, int row_start, int row_stop, int column):
    """
    Iterate over the temporal frames and compute the signal power for the specified electrode. 
    :param data: High-gamma signals of each electrode
    :param row_start: window start
    :param row_stop: window end
    :param column: electrode index
    :return: signal power
    """
    cdef int current_row = 0
    cdef double sum = 0.0
    for current_row in range(row_start, row_stop):
        sum += pow(data[current_row, column], 2)
    return sum / (row_stop - row_start)


@cython.cdivision(True)
@cython.boundscheck(False)
def compute_log_power_features(double [:, :] data, int sr, float window_length, float window_shift):
    cdef int num_windows
    cdef int win
    cdef int c
    cdef np.ndarray[np.float64_t, ndim=2] eeg_features
    cdef np.npy_intp eeg_feature_dims[2]
    cdef int start_eeg = 0
    cdef int stop_eeg = 0

    num_windows = <int> floor((data.shape[0] - window_length * sr) / (window_shift * sr)) + 1

    # Compute logarithmic high gamma broadband features
    eeg_feature_dims[0] = num_windows
    eeg_feature_dims[1] = data.shape[1]
    eeg_features = np.PyArray_SimpleNew(2, eeg_feature_dims, np.NPY_FLOAT64)
    for win in range(num_windows):
        start_eeg = int(round((win * window_shift) * sr))
        stop_eeg = int(round(start_eeg + window_length * sr))
        for c in range(data.shape[1]):
            eeg_features[win, c] = log(array_sum_and_power(data, start_eeg, stop_eeg, c) + 0.01)
    return eeg_features


cdef class WarmStartFrameBuffer:
    """
    Optimized framebuffer class for storing a window of past high-gamma activity in the computation of high-gamma
    features. The class will create new numpy arrays based on incoming high-gamma activity frames that incorporate
    a fixed context window of past high-gamma activity. If the first chunk of inserted samples does not fill a whole
    frame, it gets appended by zeros to fill the whole frame (warm start).

    Warning: This frame buffer assumes that the inserted data contains more samples than the size of the frame shift!!
    """
    cdef int frame_length_in_samples
    cdef int overlap
    cdef int nb_channels
    cdef int first_frame
    cdef double [:, :] remainder_data

    def __init__(self, float frame_length, float frame_shift, int fs, int nb_channels):
        """
        :param frame_length: Length of each frame in seconds
        :param frame_shift: Hop size in seconds
        :param fs: Sampling rate
        :param nb_channels: Number of electrodes on which the high-gamma activity was computed
        """
        cdef int frame_shift_in_samples
        cdef np.npy_intp remainder_data_dims[2]

        frame_shift_in_samples = <int> (frame_shift * fs)
        self.frame_length_in_samples = <int> (frame_length * fs)
        self.overlap = self.frame_length_in_samples - frame_shift_in_samples
        self.first_frame = True
        self.nb_channels = nb_channels

        remainder_data_dims[0] = self.overlap
        remainder_data_dims[1] = self.nb_channels
        self.remainder_data = np.PyArray_SimpleNew(2, remainder_data_dims, np.NPY_FLOAT64)

        # Zero out the remainder data
        for row in range(0, self.overlap):
            for col in range(0, self.nb_channels):
                self.remainder_data[row, col] = 0

    def reset(self):
        self.first_frame = True
        for row in range(0, self.overlap):
            for col in range(0, self.nb_channels):
                self.remainder_data[row, col] = 0

    def insert(self, double [:, :] data):
        cdef int nb_prefill_samples
        cdef np.npy_intp return_data_dims[2]
        cdef double [:, :] return_data

        return_data_dims[1] = data.shape[1]
        # CASE 1: Received first chunk of data which is has more samples than the frame buffer length:
        # -> Store overlap in remainder and return data.
        if self.first_frame and data.shape[0] >= self.frame_length_in_samples:
            self.first_frame = False
            self.remainder_data = data[-self.overlap:, :]
            return data

        # CASE 2: Received first chunk of data which is has fewer samples than the specified frame buffer length:
        # -> Pad with zeros and store overlap in remainder, before returning the frame.
        elif self.first_frame and data.shape[0] < self.frame_length_in_samples:
            nb_prefill_samples = self.frame_length_in_samples - data.shape[0]

            return_data_dims[0] = self.frame_length_in_samples
            return_data = np.PyArray_SimpleNew(2, return_data_dims, np.NPY_FLOAT64)

            return_data[0:nb_prefill_samples, :] = 0
            return_data[nb_prefill_samples:, :] = data

            self.first_frame = False
            self.remainder_data = return_data[-self.overlap:, :]
            return return_data
        else:
            return_data_dims[0] = self.overlap + data.shape[0]
            return_data = np.PyArray_SimpleNew(2, return_data_dims, np.NPY_FLOAT64)

            return_data[0:self.overlap, :] = self.remainder_data
            return_data[self.overlap:, :] = data

            self.remainder_data = return_data[-self.overlap:, :]
            return return_data
