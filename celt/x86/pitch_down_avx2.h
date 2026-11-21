#ifndef PITCH_DOWN_AVX2_H
#define PITCH_DOWN_AVX2_H

#include "avx2_guards.h"

#if defined(OPUS_CELT_AVX2)
/* x0/x1: celt_sig channels (x1 may be NULL), factor-2 decimation. */
void pitch_downsample_front2_avx2(const float *x0, const float *x1,
      float *x_lp, int len);
/* scratch must hold N+5 floats. */
void celt_fir5_avx2(float *x, const float *num, int N, float *scratch);
#endif

#endif
