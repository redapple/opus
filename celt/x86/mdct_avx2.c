/* AVX2 pre- and post-rotation loops for the float forward MDCT.

   Same licence as the rest of Opus (BSD-3-clause, see COPYING).

   Every output float is produced by the same IEEE single-precision
   operations, in the same order, as the scalar loops in mdct.c (separate
   mul and add/sub, never FMA), so the result is bit-identical. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if !defined(FIXED_POINT) && !defined(ENABLE_QEXT) && !defined(ENABLE_PFA) && defined(OPUS_X86_MAY_HAVE_AVX2)

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

#include <immintrin.h>
#include "kiss_fft.h"
#include "x86/mdct_avx2.h"

#define SWAP_RI(x) _mm256_permute_ps((x), 0xB1)

void clt_mdct_prerot_avx2(const float *f, const float *t, float scale,
      const opus_int16 *bitrev, kiss_fft_cpx *f2, int N4)
{
   int i = 0;
   const __m256 sc = _mm256_set1_ps(scale);
   for (; i + 4 <= N4; i += 4)
   {
      __m256 d = _mm256_loadu_ps(f + 2*i);        /* re0 im0 re1 im1 ... */
      __m256 w = _mm256_loadu_ps(t + 2*i);        /* t0 t1 t0 t1 ... */
      __m256 T0 = _mm256_moveldup_ps(w);
      __m256 T1 = _mm256_movehdup_ps(w);
      /* yr = re*t1 - im*t0 ; yi = im*t1 + re*t0 */
      __m256 y = _mm256_addsub_ps(_mm256_mul_ps(d, T1), _mm256_mul_ps(SWAP_RI(d), T0));
      __m128d lo, hi;
      y = _mm256_mul_ps(y, sc);
      lo = _mm256_castpd256_pd128(_mm256_castps_pd(y));
      hi = _mm256_extractf128_pd(_mm256_castps_pd(y), 1);
      _mm_storel_pd((double*)&f2[bitrev[i]],   lo);
      _mm_storeh_pd((double*)&f2[bitrev[i+1]], lo);
      _mm_storel_pd((double*)&f2[bitrev[i+2]], hi);
      _mm_storeh_pd((double*)&f2[bitrev[i+3]], hi);
   }
   for (; i < N4; i++)
   {
      float re = f[2*i], im = f[2*i+1], t0 = t[2*i], t1 = t[2*i+1];
      float yr = re*t1 - im*t0;
      float yi = im*t1 + re*t0;
      f2[bitrev[i]].r = yr*scale;
      f2[bitrev[i]].i = yi*scale;
   }
}

/* stride == 1 only.  out[2i] = yr(i), out[N2-1-2i] = yi(i), N2 = 2*N4.
   A front chunk of four indices and the mirrored back chunk are
   processed together so that every store is a contiguous 8-float write. */
void clt_mdct_postrot_avx2(const kiss_fft_cpx *fp, const float *t, float *out, int N4)
{
   int i = 0, n;
   const __m256i rev = _mm256_setr_epi32(0, 7, 0, 5, 0, 3, 0, 1);
   for (; i + 4 <= N4 - 4 - i; i += 4)
   {
      int j = N4 - 4 - i;
      __m256 df = _mm256_loadu_ps((const float*)(fp + i));
      __m256 db = _mm256_loadu_ps((const float*)(fp + j));
      __m256 wf = _mm256_loadu_ps(t + 2*i);
      __m256 wb = _mm256_loadu_ps(t + 2*j);
      /* yr = fi*t0 - fr*t1 ; yi = fr*t0 + fi*t1 */
      __m256 vf = _mm256_addsub_ps(_mm256_mul_ps(SWAP_RI(df), _mm256_moveldup_ps(wf)),
                                   _mm256_mul_ps(df, _mm256_movehdup_ps(wf)));
      __m256 vb = _mm256_addsub_ps(_mm256_mul_ps(SWAP_RI(db), _mm256_moveldup_ps(wb)),
                                   _mm256_mul_ps(db, _mm256_movehdup_ps(wb)));
      /* front out[2i..2i+7]  = yr_f(k) interleaved with reversed yi_b
         back  out[2j..2j+7]  = yr_b(k) interleaved with reversed yi_f */
      _mm256_storeu_ps(out + 2*i, _mm256_blend_ps(vf, _mm256_permutevar8x32_ps(vb, rev), 0xAA));
      _mm256_storeu_ps(out + 2*j, _mm256_blend_ps(vb, _mm256_permutevar8x32_ps(vf, rev), 0xAA));
   }
   /* Middle remainder [i, N4-i), scalar. */
   n = N4 - i;
   for (; i < n; i++)
   {
      float fr = fp[i].r, fi = fp[i].i, t0 = t[2*i], t1 = t[2*i+1];
      out[2*i]            = fi*t0 - fr*t1;
      out[2*(N4 - 1) + 1 - 2*i] = fr*t0 + fi*t1;
   }
}

#endif
