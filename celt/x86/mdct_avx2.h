#ifndef MDCT_AVX2_H
#define MDCT_AVX2_H

#include "kiss_fft.h"

#include "avx2_guards.h"
#if defined(OPUS_CELT_AVX2_MDCT)
#define MDCT_HAVE_AVX2 1
void clt_mdct_prerot_avx2(const float *f, const float *t, float scale,
      const opus_int16 *bitrev, kiss_fft_cpx *f2, int N4);
void clt_mdct_postrot_avx2(const kiss_fft_cpx *fp, const float *t, float *out, int N4);
/* Backward (inverse) MDCT helpers; bit-identical to the scalar loops. */
void clt_imdct_prerot_avx2(const float *in, const float *t, const opus_int16 *bitrev,
      kiss_fft_cpx *y, int N2, int N4);   /* stride 1 only */
int clt_imdct_postrot_avx2(float *y, const float *t, int N2, int N4); /* returns first pair index left to the scalar loop */
int clt_imdct_tdac_avx2(float *out, const float *window, int overlap); /* returns first i left to the scalar loop */
#endif

#endif
