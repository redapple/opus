#ifndef ANALYSIS_AVX2_H
#define ANALYSIS_AVX2_H

#include "opus_types.h"
#include "kiss_fft.h"

#if !defined(FIXED_POINT) && defined(OPUS_X86_MAY_HAVE_AVX2)
#define ANALYSIS_HAVE_AVX2 1
/* Vectorised version of the per-bin phase/tonality loop in tonality_analysis()
   (bins 1..N2-1, N = 2*N2 = 480). Handles bins in groups of 8 starting at 1
   and returns the first bin it did not process; the caller finishes the rest
   with the scalar loop. Bit-identical to the scalar loop. */
int tonality_bins_avx2(const kiss_fft_cpx *out, int N2, float *A, float *dA,
      float *d2A, float *tonality, float *tonality2, float *noisiness,
      float tonal_k);
#endif

#endif
