/* AVX2 versions of two float loops in celt/pitch.c: the stride-2 smoothing
   at the start of pitch_downsample() and celt_fir5().

   Same licence as the rest of Opus (BSD-3-clause, see COPYING).

   Both are bit-identical to the scalar code: every output element is
   computed with the same operations in the same order, just 8 elements at
   a time. FP contraction is disabled so no FMA is formed. */

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
#include "pitch_down_avx2.h"

/* Split 16 consecutive floats at p into their 8 even and 8 odd elements. */
static void deint16(const float *p, __m256 *even, __m256 *odd)
{
   __m256 v0 = _mm256_loadu_ps(p);
   __m256 v1 = _mm256_loadu_ps(p + 8);
   __m256 e = _mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(2, 0, 2, 0));
   __m256 o = _mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(3, 1, 3, 1));
   *even = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(e), 0xD8));
   *odd = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(o), 0xD8));
}

/* x_lp[i] = .25*x[2i-1] + .25*x[2i+1] + .5*x[2i], i = 1..len-1, evaluated
   as ((.25a + .25b) + .5c) like the scalar expression. If accumulate is
   set, the result is added to x_lp[i] (second channel). Returns the first
   index not handled. */
static int smooth2(const float *x, float *x_lp, int len, int accumulate)
{
   const __m256 q = _mm256_set1_ps(.25f);
   const __m256 h = _mm256_set1_ps(.5f);
   int i = 1;
   for (; i+8<=len; i+=8)
   {
      __m256 ev, od, odp, dummy, t;
      deint16(x + 2*i, &ev, &od);        /* x[2i+2k], x[2i+2k+1] */
      deint16(x + 2*i - 2, &dummy, &odp);/* x[2i+2k-1] */
      t = _mm256_add_ps(_mm256_mul_ps(q, odp), _mm256_mul_ps(q, od));
      t = _mm256_add_ps(t, _mm256_mul_ps(h, ev));
      if (accumulate)
         t = _mm256_add_ps(_mm256_loadu_ps(&x_lp[i]), t);
      _mm256_storeu_ps(&x_lp[i], t);
   }
   return i;
}

void pitch_downsample_front2_avx2(const float *x0, const float *x1,
      float *x_lp, int len)
{
   int i, i1;
   i = smooth2(x0, x_lp, len, 0);
   for (; i<len; i++)
      x_lp[i] = .25f*x0[(2*i-1)] + .25f*x0[(2*i+1)] + .5f*x0[2*i];
   x_lp[0] = .25f*x0[1] + .5f*x0[0];
   if (x1)
   {
      i1 = smooth2(x1, x_lp, len, 1);
      for (; i1<len; i1++)
         x_lp[i1] += .25f*x1[(2*i1-1)] + .25f*x1[(2*i1+1)] + .5f*x1[2*i1];
      x_lp[0] += .25f*x1[1] + .5f*x1[0];
   }
}

/* In-place 5-tap FIR with zero initial state:
   out[i] = (((( x[i] + n0*x[i-1]) + n1*x[i-2]) + n2*x[i-3]) + n3*x[i-4])
            + n4*x[i-5]. The caller supplies scratch of N+5 floats. */
void celt_fir5_avx2(float *x, const float *num, int N, float *scratch)
{
   int i;
   const __m256 n0 = _mm256_set1_ps(num[0]);
   const __m256 n1 = _mm256_set1_ps(num[1]);
   const __m256 n2 = _mm256_set1_ps(num[2]);
   const __m256 n3 = _mm256_set1_ps(num[3]);
   const __m256 n4 = _mm256_set1_ps(num[4]);
   float *b = scratch + 5;   /* b[-5..-1] = 0 */
   for (i=0;i<5;i++) scratch[i] = 0;
   for (i=0;i<N;i++) b[i] = x[i];
   for (i=0; i+8<=N; i+=8)
   {
      __m256 s = _mm256_loadu_ps(&b[i]);
      s = _mm256_add_ps(s, _mm256_mul_ps(n0, _mm256_loadu_ps(&b[i-1])));
      s = _mm256_add_ps(s, _mm256_mul_ps(n1, _mm256_loadu_ps(&b[i-2])));
      s = _mm256_add_ps(s, _mm256_mul_ps(n2, _mm256_loadu_ps(&b[i-3])));
      s = _mm256_add_ps(s, _mm256_mul_ps(n3, _mm256_loadu_ps(&b[i-4])));
      s = _mm256_add_ps(s, _mm256_mul_ps(n4, _mm256_loadu_ps(&b[i-5])));
      _mm256_storeu_ps(&x[i], s);
   }
   for (; i<N; i++)
   {
      float sum = b[i];
      sum = sum + num[0]*b[i-1];
      sum = sum + num[1]*b[i-2];
      sum = sum + num[2]*b[i-3];
      sum = sum + num[3]*b[i-4];
      sum = sum + num[4]*b[i-5];
      x[i] = sum;
   }
}

#endif
