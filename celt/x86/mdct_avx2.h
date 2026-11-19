#ifndef MDCT_AVX2_H
#define MDCT_AVX2_H

#include "kiss_fft.h"

#if !defined(FIXED_POINT) && !defined(ENABLE_QEXT) && !defined(ENABLE_PFA) && defined(OPUS_X86_MAY_HAVE_AVX2)
#define MDCT_HAVE_AVX2 1
void clt_mdct_prerot_avx2(const float *f, const float *t, float scale,
      const opus_int16 *bitrev, kiss_fft_cpx *f2, int N4);
void clt_mdct_postrot_avx2(const kiss_fft_cpx *fp, const float *t, float *out, int N4);
#endif

#endif
