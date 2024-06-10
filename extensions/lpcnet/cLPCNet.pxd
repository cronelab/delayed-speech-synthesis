cdef extern from "LPCNet/include/lpcnet.h":
    ctypedef struct LPCNetState:
        pass

    ctypedef struct LPCNetEncState:
        pass

    DEF NB_TOTAL_FEATURES = 36

    LPCNetState *lpcnet_create()
    int lpcnet_init(LPCNetState *lpcnet)
    void lpcnet_destroy(LPCNetState *st)
    void lpcnet_synthesize(LPCNetState *st, const float *features, short *output, int N)

    LPCNetEncState *lpcnet_encoder_create()
    int lpcnet_encoder_init(LPCNetEncState *st)
    void lpcnet_encoder_destroy(LPCNetEncState *st)
    int lpcnet_compute_features(LPCNetEncState *st, const short *pcm, float features[4][NB_TOTAL_FEATURES])
    int lpcnet_compute_single_frame_features(LPCNetEncState *st, const short *pcm, float features[NB_TOTAL_FEATURES])


cdef extern from "LPCNet/src/lpcnet_private.h":
    void decode_packet(float features[4][NB_TOTAL_FEATURES], float *vq_mem, const unsigned char buf[8])
