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

#include "x86/bands_avx2.h"

#ifdef BANDS_HAVE_AVX2

#include "x86/float_avx2.h"

void spreading_count_avx2(const celt_norm *x, int N, int tcount[3])
{
   int j;
   const __m256 vN = _mm256_set1_ps((float)N);
   const __m256 th0 = _mm256_set1_ps(.25f);
   const __m256 th1 = _mm256_set1_ps(.0625f);
   const __m256 th2 = _mm256_set1_ps(.015625f);
   /* Per-lane counts. A true comparison is all ones, i.e. -1, so the counts
      are decremented. */
   __m256i c0 = _mm256_setzero_si256();
   __m256i c1 = _mm256_setzero_si256();
   __m256i c2 = _mm256_setzero_si256();
   __m128i c;
   for (j=0;j+8<=N;j+=8)
   {
      __m256 v = _mm256_loadu_ps(x + j);
      __m256 x2N = _mm256_mul_ps(_mm256_mul_ps(v, v), vN);
      c0 = _mm256_add_epi32(c0, _mm256_castps_si256(_mm256_cmp_ps(x2N, th0, _CMP_LT_OQ)));
      c1 = _mm256_add_epi32(c1, _mm256_castps_si256(_mm256_cmp_ps(x2N, th1, _CMP_LT_OQ)));
      c2 = _mm256_add_epi32(c2, _mm256_castps_si256(_mm256_cmp_ps(x2N, th2, _CMP_LT_OQ)));
   }
   /* Sum the lanes: c = (c0, c1, c2, 0), negated. */
   c0 = _mm256_hadd_epi32(c0, c1);
   c2 = _mm256_hadd_epi32(c2, _mm256_setzero_si256());
   c0 = _mm256_hadd_epi32(c0, c2);
   c = _mm_sub_epi32(_mm_setzero_si128(),
         _mm_add_epi32(_mm256_castsi256_si128(c0), _mm256_extracti128_si256(c0, 1)));
   tcount[0] = _mm_extract_epi32(c, 0);
   tcount[1] = _mm_extract_epi32(c, 1);
   tcount[2] = _mm_extract_epi32(c, 2);
   for (;j<N;j++)
   {
      float x2N = (x[j]*x[j])*(float)N;
      tcount[0] += x2N < .25f;
      tcount[1] += x2N < .0625f;
      tcount[2] += x2N < .015625f;
   }
}

#endif
