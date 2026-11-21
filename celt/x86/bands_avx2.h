#ifndef BANDS_AVX2_H
#define BANDS_AVX2_H

#include "arch.h"

#include "avx2_guards.h"
#if defined(OPUS_CELT_AVX2_NO_QEXT)
#define BANDS_HAVE_AVX2 1
/* Rough CDF of x[j]^2*N used by spreading_decision(): tcount[k] counts the
   samples with x^2*N below 0.25, 0.0625 and 0.015625. Bit-identical to the
   scalar loop. */
void spread_count_avx2(const celt_norm *x, int N, int *tcount);
#endif

#endif
