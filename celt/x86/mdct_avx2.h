/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef MDCT_AVX2_H
#define MDCT_AVX2_H

#include "kiss_fft.h"
#include "x86cpu.h"

#if defined(OPUS_X86_MAY_HAVE_AVX2) && !defined(FIXED_POINT) && \
    !defined(ENABLE_QEXT) && !defined(ENABLE_PFA)
#define MDCT_HAVE_AVX2

/* Each function replaces one loop of clt_mdct_forward_c() or
   clt_mdct_backward_c() (see the comments in mdct.c) and is bit-exact with
   it. The ones taking no stride argument only work for stride==1. */

void clt_mdct_prerot_avx2(const float *f, const float *t, float scale,
      const opus_int16 *bitrev, kiss_fft_cpx *f2, int N4);

void clt_mdct_postrot_avx2(const kiss_fft_cpx *fp, const float *t,
      float *out, int N4);

void clt_imdct_prerot_avx2(const float *in, const float *t,
      const opus_int16 *bitrev, kiss_fft_cpx *y, int N4);

void clt_imdct_postrot_avx2(float *y, const float *t, int N4);

void clt_imdct_tdac_avx2(float *out, const float *window, int overlap);
#endif

#endif
