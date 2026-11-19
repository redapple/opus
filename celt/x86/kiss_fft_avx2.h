#ifndef KISS_FFT_AVX2_H
#define KISS_FFT_AVX2_H

#include "kiss_fft.h"

#if !defined(FIXED_POINT) && defined(OPUS_X86_MAY_HAVE_AVX2)
/* Runtime check; defined in kiss_fft.c (built without AVX flags). */
int kiss_fft_use_avx2(void);
void kf_bfly2_avx2(kiss_fft_cpx *Fout, int N);
void kf_bfly3_avx2(kiss_fft_cpx *Fout, const size_t fstride,
      const kiss_fft_state *st, int m, int N, int mm);
void kf_bfly4_avx2(kiss_fft_cpx *Fout, const size_t fstride,
      const kiss_fft_state *st, int m, int N, int mm);
void kf_bfly5_avx2(kiss_fft_cpx *Fout, const size_t fstride,
      const kiss_fft_state *st, int m, int N, int mm);
#endif

#endif
