/* AVX2 helper for spreading_decision().

   Same licence as the rest of Opus (BSD-3-clause, see COPYING).

   Computes (x*x)*N with the same two IEEE single-precision multiplies as the
   scalar code and counts the comparisons, so the result is identical. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "avx2_guards.h"
#if defined(OPUS_CELT_AVX2_NO_QEXT)

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

/* One Haar butterfly on a pair of scalars, same operations as haar1(). */
static void haar1_pair(float *a, float *b)
{
   float t1 = .70710678f * *a;
   float t2 = .70710678f * *b;
   *a = t1 + t2;
   *b = t1 - t2;
}

void haar1_avx2(celt_norm *X, int N0, int stride)
{
   const __m256 vc = _mm256_set1_ps(.70710678f);
   int i, j;
   N0 >>= 1;
   if (stride >= 8 && (stride & 7) == 0)
   {
      /* Independent columns: 8 at a time. */
      for (j = 0; j < N0; j++)
      {
         float *a = X + stride*2*j;
         float *b = a + stride;
         for (i = 0; i < stride; i += 8)
         {
            __m256 t1 = _mm256_mul_ps(vc, _mm256_loadu_ps(a+i));
            __m256 t2 = _mm256_mul_ps(vc, _mm256_loadu_ps(b+i));
            _mm256_storeu_ps(a+i, _mm256_add_ps(t1, t2));
            _mm256_storeu_ps(b+i, _mm256_sub_ps(t1, t2));
         }
      }
   }
   else if (stride == 4)
   {
      /* a(4) | b(4) in one register. */
      for (j = 0; j < N0; j++)
      {
         __m256 v = _mm256_mul_ps(vc, _mm256_loadu_ps(X + 8*j));
         __m256 sw = _mm256_permute2f128_ps(v, v, 1);
         __m256 sum = _mm256_add_ps(v, sw);
         __m256 dif = _mm256_sub_ps(sw, v);
         _mm256_storeu_ps(X + 8*j, _mm256_blend_ps(sum, dif, 0xF0));
      }
   }
   else if (stride == 2)
   {
      /* Two pairs per register: a0 a1 b0 b1 | a0' a1' b0' b1'. */
      for (j = 0; j+2 <= N0; j += 2)
      {
         __m256 v = _mm256_mul_ps(vc, _mm256_loadu_ps(X + 4*j));
         __m256 sw = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2));
         __m256 sum = _mm256_add_ps(v, sw);
         __m256 dif = _mm256_sub_ps(sw, v);
         _mm256_storeu_ps(X + 4*j, _mm256_blend_ps(sum, dif, 0xCC));
      }
      for (; j < N0; j++)
      {
         haar1_pair(X + 4*j, X + 4*j + 2);
         haar1_pair(X + 4*j + 1, X + 4*j + 3);
      }
   }
   else if (stride == 1)
   {
      /* Four adjacent pairs per register. */
      for (j = 0; j+4 <= N0; j += 4)
      {
         __m256 v = _mm256_mul_ps(vc, _mm256_loadu_ps(X + 2*j));
         __m256 e = _mm256_moveldup_ps(v);
         __m256 o = _mm256_movehdup_ps(v);
         __m256 sum = _mm256_add_ps(e, o);
         __m256 dif = _mm256_sub_ps(e, o);
         _mm256_storeu_ps(X + 2*j, _mm256_blend_ps(sum, dif, 0xAA));
      }
      for (; j < N0; j++)
         haar1_pair(X + 2*j, X + 2*j + 1);
   }
   else
   {
      for (i = 0; i < stride; i++)
         for (j = 0; j < N0; j++)
            haar1_pair(X + stride*2*j + i, X + stride*(2*j+1) + i);
   }
}

#endif
