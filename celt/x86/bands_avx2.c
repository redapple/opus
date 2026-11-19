/* AVX2 helper for spreading_decision().

   Same licence as the rest of Opus (BSD-3-clause, see COPYING).

   Computes (x*x)*N with the same two IEEE single-precision multiplies as the
   scalar code and counts the comparisons, so the result is identical. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if !defined(FIXED_POINT) && !defined(ENABLE_QEXT) && defined(OPUS_X86_MAY_HAVE_AVX2)

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

#include <immintrin.h>
#include "opus_types.h"
#include "opus_defines.h"
#include "x86/bands_avx2.h"

void spread_count_avx2(const celt_norm *x, int N, int *tcount)
{
   int j = 0;
   int c0 = 0, c1 = 0, c2 = 0;
   const __m256 vN = _mm256_set1_ps((float)N);
   const __m256 t0 = _mm256_set1_ps(0.25f);
   const __m256 t1 = _mm256_set1_ps(0.0625f);
   const __m256 t2 = _mm256_set1_ps(0.015625f);
   for (; j+8 <= N; j += 8)
   {
      __m256 v = _mm256_loadu_ps(&x[j]);
      __m256 x2N = _mm256_mul_ps(_mm256_mul_ps(v, v), vN);
      c0 += _mm_popcnt_u32(_mm256_movemask_ps(_mm256_cmp_ps(x2N, t0, _CMP_LT_OQ)));
      c1 += _mm_popcnt_u32(_mm256_movemask_ps(_mm256_cmp_ps(x2N, t1, _CMP_LT_OQ)));
      c2 += _mm_popcnt_u32(_mm256_movemask_ps(_mm256_cmp_ps(x2N, t2, _CMP_LT_OQ)));
   }
   for (; j < N; j++)
   {
      float x2N = (x[j]*x[j])*(float)N;
      if (x2N < 0.25f) c0++;
      if (x2N < 0.0625f) c1++;
      if (x2N < 0.015625f) c2++;
   }
   tcount[0] = c0;
   tcount[1] = c1;
   tcount[2] = c2;
}

#endif
