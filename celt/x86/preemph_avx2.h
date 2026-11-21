#ifndef PREEMPH_AVX2_H
#define PREEMPH_AVX2_H

#include "avx2_guards.h"

#if defined(OPUS_CELT_AVX2)
/* CC must be 1 or 2; N >= 9. pcmp points at this channel's first sample. */
void celt_preemphasis_fast_avx2(const float *pcmp, float *inp, int N, int CC,
      float coef0, float *mem);
#endif

#endif
