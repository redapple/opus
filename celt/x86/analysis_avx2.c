/* AVX2 version of the per-bin phase/tonality loop of the music/speech
   analysis.

   Same licence as the rest of Opus (BSD-3-clause, see COPYING).

   Every lane goes through the same IEEE single-precision operations, in the
   same order, as the scalar code in analysis.c (and fast_atan2f() in
   mathops.h), never FMA, so the result is bit-identical. */

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
#include <math.h>
#include "opus_defines.h"
#include "arch.h"
#include "mathops.h"
#include "x86/analysis_avx2.h"

/* De-interleave 8 consecutive complex values starting at p into re/im, in
   ascending index order. */
static OPUS_INLINE void load_cpx8(const kiss_fft_cpx *p, __m256 *re, __m256 *im)
{
   __m256 a = _mm256_loadu_ps((const float*)p);
   __m256 b = _mm256_loadu_ps((const float*)p + 8);
   *re = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(_mm256_shuffle_ps(a, b, 0x88)), 0xD8));
   *im = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(_mm256_shuffle_ps(a, b, 0xDD)), 0xD8));
}

static OPUS_INLINE __m256 reverse8(__m256 v)
{
   return _mm256_permutevar8x32_ps(v, _mm256_setr_epi32(7,6,5,4,3,2,1,0));
}

/* Same as fast_atan2f() in mathops.h. */
static OPUS_INLINE __m256 fast_atan2f_avx2(__m256 y, __m256 x)
{
   const __m256 cA = _mm256_set1_ps(0.43157974f);
   const __m256 cB = _mm256_set1_ps(0.67848403f);
   const __m256 cC = _mm256_set1_ps(0.08595542f);
   const __m256 cE = _mm256_set1_ps((float)PI/2);
   const __m256 ncE = _mm256_set1_ps(-(float)PI/2);
   const __m256 zero = _mm256_setzero_ps();
   const __m256 signmask = _mm256_set1_ps(-0.f);
   __m256 x2 = _mm256_mul_ps(x, x);
   __m256 y2 = _mm256_mul_ps(y, y);
   __m256 xy = _mm256_mul_ps(x, y);
   __m256 small = _mm256_cmp_ps(_mm256_add_ps(x2, y2), _mm256_set1_ps(1e-18f), _CMP_LT_OQ);
   __m256 use_y = _mm256_cmp_ps(x2, y2, _CMP_LT_OQ);
   __m256 sel_y = _mm256_blendv_ps(cE, ncE, _mm256_cmp_ps(y, zero, _CMP_LT_OQ));
   /* x2 < y2 branch: -x*y*(y2 + cA*x2) / den + (y<0 ? -cE : cE) */
   __m256 den1 = _mm256_mul_ps(_mm256_add_ps(y2, _mm256_mul_ps(cB, x2)),
                               _mm256_add_ps(y2, _mm256_mul_ps(cC, x2)));
   __m256 nx = _mm256_xor_ps(x, signmask);
   __m256 r1 = _mm256_add_ps(_mm256_div_ps(_mm256_mul_ps(_mm256_mul_ps(nx, y),
                  _mm256_add_ps(y2, _mm256_mul_ps(cA, x2))), den1), sel_y);
   /* else branch: x*y*(x2 + cA*y2) / den + (y<0 ? -cE : cE) - (x*y<0 ? -cE : cE) */
   __m256 den2 = _mm256_mul_ps(_mm256_add_ps(x2, _mm256_mul_ps(cB, y2)),
                               _mm256_add_ps(x2, _mm256_mul_ps(cC, y2)));
   __m256 sel_xy = _mm256_blendv_ps(cE, ncE, _mm256_cmp_ps(xy, zero, _CMP_LT_OQ));
   __m256 r2 = _mm256_sub_ps(_mm256_add_ps(_mm256_div_ps(_mm256_mul_ps(xy,
                  _mm256_add_ps(x2, _mm256_mul_ps(cA, y2))), den2), sel_y), sel_xy);
   __m256 r = _mm256_blendv_ps(r2, r1, use_y);
   return _mm256_andnot_ps(small, r);
}

/* x - (float)float2int(x), round-to-nearest-even. */
static OPUS_INLINE __m256 frac_part(__m256 x)
{
   return _mm256_sub_ps(x, _mm256_cvtepi32_ps(_mm256_cvtps_epi32(x)));
}

int tonality_bins_avx2(const kiss_fft_cpx *out, int N2, float *A, float *dA,
      float *d2A, float *tonality, float *tonality2, float *noisiness,
      float tonal_k)
{
   const int N = 2*N2;
   const __m256 norm = _mm256_set1_ps((float)(.5f/M_PI));
   const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
   const __m256 one = _mm256_set1_ps(1.f);
   const __m256 k = _mm256_set1_ps(tonal_k);
   const __m256 off = _mm256_set1_ps(.015f);
   const __m256 quarter = _mm256_set1_ps(.25f);
   const __m256 two = _mm256_set1_ps(2.f);
   int i;
   for (i=1; i+8 <= N2; i+=8)
   {
      __m256 or_, oi, pr, pi_;
      load_cpx8(&out[i], &or_, &oi);
      load_cpx8(&out[N-i-7], &pr, &pi_);   /* ascending N-i-7 .. N-i */
      pr = reverse8(pr);                   /* lane l -> out[N-(i+l)] */
      pi_ = reverse8(pi_);
      __m256 X1r = _mm256_add_ps(or_, pr);
      __m256 X1i = _mm256_sub_ps(oi, pi_);
      __m256 X2r = _mm256_add_ps(oi, pi_);
      __m256 X2i = _mm256_sub_ps(pr, or_);

      __m256 angle = _mm256_mul_ps(norm, fast_atan2f_avx2(X1i, X1r));
      __m256 vA = _mm256_loadu_ps(&A[i]);
      __m256 vdA = _mm256_loadu_ps(&dA[i]);
      __m256 vd2A = _mm256_loadu_ps(&d2A[i]);
      __m256 d_angle = _mm256_sub_ps(angle, vA);
      __m256 d2_angle = _mm256_sub_ps(d_angle, vdA);

      __m256 angle2 = _mm256_mul_ps(norm, fast_atan2f_avx2(X2i, X2r));
      __m256 d_angle2 = _mm256_sub_ps(angle2, angle);
      __m256 d2_angle2 = _mm256_sub_ps(d_angle2, d_angle);

      __m256 mod1 = frac_part(d2_angle);
      __m256 nz = _mm256_and_ps(mod1, absmask);
      mod1 = _mm256_mul_ps(mod1, mod1);
      mod1 = _mm256_mul_ps(mod1, mod1);

      __m256 mod2 = frac_part(d2_angle2);
      nz = _mm256_add_ps(nz, _mm256_and_ps(mod2, absmask));
      mod2 = _mm256_mul_ps(mod2, mod2);
      mod2 = _mm256_mul_ps(mod2, mod2);

      __m256 avg_mod = _mm256_mul_ps(quarter,
            _mm256_add_ps(_mm256_add_ps(vd2A, mod1), _mm256_mul_ps(two, mod2)));
      _mm256_storeu_ps(&noisiness[i], nz);
      _mm256_storeu_ps(&tonality[i], _mm256_sub_ps(
            _mm256_div_ps(one, _mm256_add_ps(one, _mm256_mul_ps(k, avg_mod))), off));
      _mm256_storeu_ps(&tonality2[i], _mm256_sub_ps(
            _mm256_div_ps(one, _mm256_add_ps(one, _mm256_mul_ps(k, mod2))), off));
      _mm256_storeu_ps(&A[i], angle2);
      _mm256_storeu_ps(&dA[i], d_angle2);
      _mm256_storeu_ps(&d2A[i], mod2);
   }
   return i;
}

#endif
