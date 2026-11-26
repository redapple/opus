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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "x86/mathops_avx2.h"

#if defined(OPUS_X86_MAY_HAVE_AVX2) && !defined(FIXED_POINT)

#include "x86/float_avx2.h"

float celt_maxabs16_avx2(const float *x, int len)
{
   int i;
   float maxval, minval;
   float tmp[8];
   __m256 vmax = _mm256_setzero_ps();
   __m256 vmin = _mm256_setzero_ps();
   /* max_ps(a, b) and min_ps(a, b) are a > b ? a : b and a < b ? a : b, like
      MAX16() and MIN16(). */
   for (i=0;i+8<=len;i+=8)
   {
      __m256 v = _mm256_loadu_ps(x + i);
      vmax = _mm256_max_ps(vmax, v);
      vmin = _mm256_min_ps(vmin, v);
   }
   _mm256_storeu_ps(tmp, _mm256_max_ps(vmax, _mm256_sub_ps(_mm256_setzero_ps(), vmin)));
   maxval = MAX16(MAX16(MAX16(tmp[0], tmp[1]), MAX16(tmp[2], tmp[3])),
                  MAX16(MAX16(tmp[4], tmp[5]), MAX16(tmp[6], tmp[7])));
   minval = 0;
   for (;i<len;i++)
   {
      maxval = MAX16(maxval, x[i]);
      minval = MIN16(minval, x[i]);
   }
   return MAX16(maxval, -minval);
}

#endif
