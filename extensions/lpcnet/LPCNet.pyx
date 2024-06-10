cimport cLPCNet
import numpy as np
cimport numpy as np


cdef class LPCNet:
    """
    The LPCNet class wraps functionality for synthesizing audio using the LPCNet decoder
    """
    LPCNET_FRAME_SIZE: int = 160

    cdef cLPCNet.LPCNetState *_c_state

    def __cinit__(self):
        self._c_state = cLPCNet.lpcnet_create()
        if self._c_state is NULL:
            raise MemoryError

    def __dealloc__(self):
        if self._c_state is not NULL:
            cLPCNet.lpcnet_destroy(self._c_state)

    def reset_decoder(self):
        """
        Calls the lpc_init c function which zero's out the internal memory by using a memset call and assigns default
        values for some internal variables (sampling logit table and rng)
        """
        cLPCNet.lpcnet_init(self._c_state)

    def synthesize(self, np.ndarray[np.float32_t, ndim=1] features):
        """
        Generate audio for the provided LPC features by calling the lpcnet_synthesize c function
        :param features: numpy array of LPC features
        :return: numpy array of 160 (10 ms) acoustic samples
        """
        cdef np.ndarray[np.int16_t, ndim=1] result = np.ones(self.LPCNET_FRAME_SIZE, dtype=np.int16, order='C')

        # 10 ms with 16kHz equals 160 acoustic samples
        cLPCNet.lpcnet_synthesize(self._c_state, <float *> features.data, <short *> result.data, self.LPCNET_FRAME_SIZE)
        return result


cdef class LPCFeatureEncoder:
    """
    The LPCFeatureEncoder class wraps functionality for generating LPC features for the decoder
    """
    NB_FEATURES: int = 20
    NB_TOTAL_FEATURES: int = 36
    LPCNET_FRAME_SIZE: int = 160

    cdef cLPCNet.LPCNetEncState *_c_state

    def __cinit__(self):
        self._c_state = cLPCNet.lpcnet_encoder_create()
        if self._c_state is NULL:
            raise MemoryError

    def __dealloc__(self):
        if self._c_state is not NULL:
            cLPCNet.lpcnet_encoder_destroy(self._c_state)

    def reset_encoder(self):
        cLPCNet.lpcnet_encoder_init(self._c_state)

    def compute_LPC_features(self, np.ndarray[np.int16_t, ndim=1] audio_samples):
        """
        Generate LPC features audio for the provided audio by calling the lpcnet_compute_single_frame_features c
        function. The sample rate of the audio samples has to be 16k to meet the specification of the LPCNet vocoder
        :param audio_samples: Numpy array of audio samples in PCM int16 format.
        :return: numpy array of LPC features
        """
        cdef int num_feature_frames = len(audio_samples) // self.LPCNET_FRAME_SIZE
        cdef np.ndarray[np.int16_t, ndim=1] frame = np.zeros(self.LPCNET_FRAME_SIZE, dtype=np.int16)
        cdef np.ndarray[np.float32_t, ndim=1] features = np.zeros(self.NB_TOTAL_FEATURES, dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros((num_feature_frames, self.NB_FEATURES),
                                                                dtype=np.float32)
        cdef int start = 0
        cdef int end = 0

        for i in range(num_feature_frames):
            start = i * self.LPCNET_FRAME_SIZE
            end = start + self.LPCNET_FRAME_SIZE
            frame = audio_samples[start:end]
            cLPCNet.lpcnet_compute_single_frame_features(self._c_state, <short *> frame.data, <float *> features.data)
            result[i] = features[:self.NB_FEATURES]

        return result


class LPCFeatureFile:
    """
    The LPCNet feature file class wraps a *.f32 feature file generated from the lpcnet_demo application:
    lpcnet_demo -features <input.pcm> <features.f32>

    This class can be used to iterate over all frames (with the possibility to loop infinitely).
    """
    def __init__(self, filename, loop=False, nb_total_features=36):
        with open(filename, 'rb') as file_handler:
            raw = file_handler.read()

        self.features = np.frombuffer(raw, dtype=np.float32).reshape((-1, nb_total_features))
        self.index = 0
        self.loop = loop

    def __iter__(self):
        return self

    def __next__(self):
        try:
            features = self.features[self.index]
        except IndexError: raise StopIteration

        self.index += 1
        if self.index == len(self.features) and self.loop: self.index = 0
        return features[0:20]
