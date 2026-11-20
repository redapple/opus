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

#include "x86/analysis_avx2.h"

#ifdef ANALYSIS_HAVE_AVX2

#include "x86/float_avx2.h"
#include "arch.h"
#include "mathops.h"

/* Same value as in analysis.c. */
#ifndef M_PI
#define M_PI 3.141592653
#endif

/* De-interleaves the complex values p[0..7] into real and imaginary parts. */
static OPUS_INLINE void load_cpx8(const kiss_fft_cpx *p, __m256 *re, __m256 *im)
{
   const __m256i idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
   /* a = (re0..re3, im0..im3) and b = (re4..re7, im4..im7) */
   __m256 a = _mm256_permutevar8x32_ps(_mm256_loadu_ps((const float*)p), idx);
   __m256 b = _mm256_permutevar8x32_ps(_mm256_loadu_ps((const float*)(p + 4)), idx);
   *re = _mm256_permute2f128_ps(a, b, 0x20);
   *im = _mm256_permute2f128_ps(a, b, 0x31);
}

static OPUS_INLINE __m256 reverse8(__m256 x)
{
   return _mm256_permutevar8x32_ps(x, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
}

static OPUS_INLINE __m256 neg(__m256 x)
{
   return _mm256_xor_ps(x, _mm256_set1_ps(-0.f));
}

/* tansig_approx() and sigmoid_approx() from mlp.c. min/max have the same
   semantics as MIN32()/MAX32() with the operands in the same order. */
static OPUS_INLINE __m256 tansig_avx2(__m256 x)
{
   const __m256 N0 = _mm256_set1_ps(952.52801514f);
   const __m256 N1 = _mm256_set1_ps(96.39235687f);
   const __m256 N2 = _mm256_set1_ps(0.60863042f);
   const __m256 D0 = _mm256_set1_ps(952.72399902f);
   const __m256 D1 = _mm256_set1_ps(413.36801147f);
   const __m256 D2 = _mm256_set1_ps(11.88600922f);
   __m256 X2, num, den;
   X2 = _mm256_mul_ps(x, x);
   num = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(N2, X2), N1), X2), N0);
   den = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(D2, X2), D1), X2), D0);
   num = _mm256_div_ps(_mm256_mul_ps(num, x), den);
   return _mm256_max_ps(_mm256_set1_ps(-1.f), _mm256_min_ps(_mm256_set1_ps(1.f), num));
}

static OPUS_INLINE __m256 sigmoid_avx2(__m256 x)
{
   const __m256 half = _mm256_set1_ps(.5f);
   return _mm256_add_ps(half, _mm256_mul_ps(half, tansig_avx2(_mm256_mul_ps(half, x))));
}

static OPUS_INLINE __m256 load_int8(const opus_int8 *p)
{
   return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(const void*)p)));
}

/* WEIGHTS_SCALE*(bias[k] + sum_j weights[j*col_stride+k]*x[j]
                   + sum_j rweights[j*col_stride+k]*rx[j]) for k = 0..7,
   with the sums accumulated in the same order as gemm_accum() calls in
   mlp.c. The second sum (recurrent weights) may be empty. */
static OPUS_INLINE __m256 gemm8(const opus_int8 *bias, const opus_int8 *weights,
      int cols, int col_stride, const float *x,
      const opus_int8 *rweights, int rcols, const float *rx)
{
   int j;
   __m256 acc = load_int8(bias);
   for (j=0;j<cols;j++)
      acc = _mm256_add_ps(acc, _mm256_mul_ps(load_int8(weights + j*col_stride), _mm256_set1_ps(x[j])));
   for (j=0;j<rcols;j++)
      acc = _mm256_add_ps(acc, _mm256_mul_ps(load_int8(rweights + j*col_stride), _mm256_set1_ps(rx[j])));
   return _mm256_mul_ps(acc, _mm256_set1_ps(WEIGHTS_SCALE));
}

void analysis_compute_dense_avx2(const AnalysisDenseLayer *layer, float *output, const float *input)
{
   int i;
   int N = layer->nb_neurons;
   for (i=0;i<N;i+=8)
   {
      __m256 y = gemm8(&layer->bias[i], &layer->input_weights[i], layer->nb_inputs, N, input,
            NULL, 0, NULL);
      y = layer->sigmoid ? sigmoid_avx2(y) : tansig_avx2(y);
      _mm256_storeu_ps(&output[i], y);
   }
}

void analysis_compute_gru_avx2(const AnalysisGRULayer *gru, float *state, const float *input)
{
   int i;
   int N = gru->nb_neurons;
   int M = gru->nb_inputs;
   int stride = 3*N;
   float z[MAX_NEURONS];
   float tmp[MAX_NEURONS];
   float h[MAX_NEURONS];
   /* Update and reset gates. */
   for (i=0;i<N;i+=8)
   {
      __m256 r;
      _mm256_storeu_ps(&z[i], sigmoid_avx2(gemm8(&gru->bias[i], &gru->input_weights[i], M, stride,
            input, &gru->recurrent_weights[i], N, state)));
      r = sigmoid_avx2(gemm8(&gru->bias[N+i], &gru->input_weights[N+i], M, stride,
            input, &gru->recurrent_weights[N+i], N, state));
      _mm256_storeu_ps(&tmp[i], _mm256_mul_ps(_mm256_loadu_ps(&state[i]), r));
   }
   /* Output, which needs all of tmp = state*r. */
   for (i=0;i<N;i+=8)
   {
      __m256 zi = _mm256_loadu_ps(&z[i]);
      __m256 hi = tansig_avx2(gemm8(&gru->bias[2*N+i], &gru->input_weights[2*N+i], M, stride,
            input, &gru->recurrent_weights[2*N+i], N, tmp));
      hi = _mm256_add_ps(_mm256_mul_ps(zi, _mm256_loadu_ps(&state[i])),
            _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.f), zi), hi));
      _mm256_storeu_ps(&h[i], hi);
   }
   OPUS_COPY(state, h, N);
}

/* fast_atan2f() from mathops.h, computing both branches. */
static OPUS_INLINE __m256 fast_atan2f_avx2(__m256 y, __m256 x)
{
   const __m256 cA = _mm256_set1_ps(0.43157974f);
   const __m256 cB = _mm256_set1_ps(0.67848403f);
   const __m256 cC = _mm256_set1_ps(0.08595542f);
   const __m256 cE = _mm256_set1_ps((float)PI/2);
   const __m256 zero = _mm256_setzero_ps();
   __m256 x2, y2, xy, num, den, r1, r2, sign_y, sign_xy;
   x2 = _mm256_mul_ps(x, x);
   y2 = _mm256_mul_ps(y, y);
   xy = _mm256_mul_ps(x, y);
   /* (y<0 ? -cE : cE) and (x*y<0 ? -cE : cE) */
   sign_y = _mm256_blendv_ps(cE, neg(cE), _mm256_cmp_ps(y, zero, _CMP_LT_OQ));
   sign_xy = _mm256_blendv_ps(cE, neg(cE), _mm256_cmp_ps(xy, zero, _CMP_LT_OQ));
   /* x2<y2: -x*y*(y2 + cA*x2) / den + sign_y */
   den = _mm256_mul_ps(_mm256_add_ps(y2, _mm256_mul_ps(cB, x2)), _mm256_add_ps(y2, _mm256_mul_ps(cC, x2)));
   num = _mm256_mul_ps(_mm256_mul_ps(neg(x), y), _mm256_add_ps(y2, _mm256_mul_ps(cA, x2)));
   r1 = _mm256_add_ps(_mm256_div_ps(num, den), sign_y);
   /* else: x*y*(x2 + cA*y2) / den + sign_y - sign_xy */
   den = _mm256_mul_ps(_mm256_add_ps(x2, _mm256_mul_ps(cB, y2)), _mm256_add_ps(x2, _mm256_mul_ps(cC, y2)));
   num = _mm256_mul_ps(xy, _mm256_add_ps(x2, _mm256_mul_ps(cA, y2)));
   r2 = _mm256_sub_ps(_mm256_add_ps(_mm256_div_ps(num, den), sign_y), sign_xy);
   r1 = _mm256_blendv_ps(r2, r1, _mm256_cmp_ps(x2, y2, _CMP_LT_OQ));
   /* 0 when x2 + y2 < 1e-18f. */
   return _mm256_andnot_ps(_mm256_cmp_ps(_mm256_add_ps(x2, y2), _mm256_set1_ps(1e-18f), _CMP_LT_OQ), r1);
}

/* x - (float)float2int(x). The conversion rounds to nearest like
   float2int(). */
static OPUS_INLINE __m256 frac_avx2(__m256 x)
{
   return _mm256_sub_ps(x, _mm256_cvtepi32_ps(_mm256_cvtps_epi32(x)));
}

int tonality_bins_avx2(const kiss_fft_cpx *out, int N2, float *A, float *dA,
      float *d2A, float *tonality, float *tonality2, float *noisiness,
      float tonality_scale)
{
   int i;
   const int N = 2*N2;
   const __m256 norm = _mm256_set1_ps((float)(.5f/M_PI));
   const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
   const __m256 one = _mm256_set1_ps(1.f);
   const __m256 k = _mm256_set1_ps(tonality_scale);
   for (i=1;i+8<=N2;i+=8)
   {
      __m256 r, im, rr, rim;
      __m256 X1r, X1i, X2r, X2i;
      __m256 angle, d_angle, d2_angle, angle2, d_angle2, d2_angle2;
      __m256 mod1, mod2, avg_mod;
      /* out[i+l] and out[N-(i+l)] for l = 0..7 */
      load_cpx8(&out[i], &r, &im);
      load_cpx8(&out[N-i-7], &rr, &rim);
      rr = reverse8(rr);
      rim = reverse8(rim);
      X1r = _mm256_add_ps(r, rr);
      X1i = _mm256_sub_ps(im, rim);
      X2r = _mm256_add_ps(im, rim);
      X2i = _mm256_sub_ps(rr, r);

      angle = _mm256_mul_ps(norm, fast_atan2f_avx2(X1i, X1r));
      d_angle = _mm256_sub_ps(angle, _mm256_loadu_ps(&A[i]));
      d2_angle = _mm256_sub_ps(d_angle, _mm256_loadu_ps(&dA[i]));

      angle2 = _mm256_mul_ps(norm, fast_atan2f_avx2(X2i, X2r));
      d_angle2 = _mm256_sub_ps(angle2, angle);
      d2_angle2 = _mm256_sub_ps(d_angle2, d_angle);

      mod1 = frac_avx2(d2_angle);
      mod2 = frac_avx2(d2_angle2);
      _mm256_storeu_ps(&noisiness[i], _mm256_add_ps(_mm256_and_ps(mod1, abs_mask),
            _mm256_and_ps(mod2, abs_mask)));
      mod1 = _mm256_mul_ps(mod1, mod1);
      mod1 = _mm256_mul_ps(mod1, mod1);
      mod2 = _mm256_mul_ps(mod2, mod2);
      mod2 = _mm256_mul_ps(mod2, mod2);

      /* .25f*(d2A[i]+mod1+2*mod2) */
      avg_mod = _mm256_mul_ps(_mm256_set1_ps(.25f), _mm256_add_ps(
            _mm256_add_ps(_mm256_loadu_ps(&d2A[i]), mod1), _mm256_mul_ps(_mm256_set1_ps(2.f), mod2)));
      _mm256_storeu_ps(&tonality[i], _mm256_sub_ps(_mm256_div_ps(one,
            _mm256_add_ps(one, _mm256_mul_ps(k, avg_mod))), _mm256_set1_ps(.015f)));
      _mm256_storeu_ps(&tonality2[i], _mm256_sub_ps(_mm256_div_ps(one,
            _mm256_add_ps(one, _mm256_mul_ps(k, mod2))), _mm256_set1_ps(.015f)));

      _mm256_storeu_ps(&A[i], angle2);
      _mm256_storeu_ps(&dA[i], d_angle2);
      _mm256_storeu_ps(&d2A[i], mod2);
   }
   return i;
}

#endif
