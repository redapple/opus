/* AVX2 fast path of celt_preemphasis() (float build, no clipping, 48 kHz).

   Same licence as the rest of Opus (BSD-3-clause, see COPYING).

   The scalar loop is
       x = CELT_SIG_SCALE*pcm[CC*i];  inp[i] = x - m;  m = coef0*x;
   m depends only on the previous input sample, not on the previous m, so
   there is no recurrence and every output can be computed independently
   with the same two operations. Bit-identical to the scalar code (FP
   contraction is disabled so no FMA is formed). */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "avx2_guards.h"
#if defined(OPUS_CELT_AVX2)

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

#include <immintrin.h>
#include "arch.h"
#include "preemph_avx2.h"

/* Even-indexed floats of 16 consecutive values at p, in order. */
static __m256 even16(const float *p)
{
   __m256 v0 = _mm256_loadu_ps(p);
   __m256 v1 = _mm256_loadu_ps(p + 8);
   __m256 e = _mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(2, 0, 2, 0));
   return _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(e), 0xD8));
}

void celt_preemphasis_fast_avx2(const float *pcmp, float *inp, int N, int CC,
      float coef0, float *mem)
{
   const __m256 scale = _mm256_set1_ps(CELT_SIG_SCALE);
   const __m256 c0 = _mm256_set1_ps(coef0);
   const __m256i rot = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
   float m = *mem;
   float carry = 0;   /* x[i-1] at the start of each block */
   int i = 0;
   /* Strict "<": for CC==2 a block reads up to pcmp[2i+15]. */
   for (; i+8<N; i+=8)
   {
      __m256 p, x, prev, mv;
      p = (CC == 1) ? _mm256_loadu_ps(pcmp + i) : even16(pcmp + 2*i);
      x = _mm256_mul_ps(scale, p);
      prev = _mm256_blend_ps(_mm256_permutevar8x32_ps(x, rot),
                             _mm256_set1_ps(carry), 0x01);
      mv = _mm256_mul_ps(c0, prev);
      if (i == 0)   /* first output uses the saved filter memory */
         mv = _mm256_blend_ps(mv, _mm256_set1_ps(m), 0x01);
      _mm256_storeu_ps(inp + i, _mm256_sub_ps(x, mv));
      {
         __m128 hi = _mm256_extractf128_ps(x, 1);
         carry = _mm_cvtss_f32(_mm_shuffle_ps(hi, hi, _MM_SHUFFLE(3, 3, 3, 3)));
      }
   }
   if (i != 0)
      m = coef0*carry;
   for (; i<N; i++)
   {
      float x = CELT_SIG_SCALE*pcmp[CC*i];
      inp[i] = x - m;
      m = coef0*x;
   }
   *mem = m;
}

#endif
