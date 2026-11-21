/* AVX2 version of the music/speech analysis MLP (dense + GRU layers).

   Same licence as the rest of Opus (BSD-3-clause, see COPYING).

   The vectorisation is across output neurons: each output element goes
   through exactly the same IEEE single-precision operations, in the same
   order, as the scalar code in mlp.c (separate mul and add, never FMA), so
   the result is bit-identical. */

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
#include "opus_types.h"
#include "opus_defines.h"
#include "arch.h"
#include "x86/mlp_avx2.h"

static OPUS_INLINE __m256 tansig_avx2(__m256 x)
{
   const __m256 N0 = _mm256_set1_ps(952.52801514f);
   const __m256 N1 = _mm256_set1_ps(96.39235687f);
   const __m256 N2 = _mm256_set1_ps(0.60863042f);
   const __m256 D0 = _mm256_set1_ps(952.72399902f);
   const __m256 D1 = _mm256_set1_ps(413.36801147f);
   const __m256 D2 = _mm256_set1_ps(11.88600922f);
   __m256 X2 = _mm256_mul_ps(x, x);
   __m256 num = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(N2, X2), N1), X2), N0);
   __m256 den = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(D2, X2), D1), X2), D0);
   num = _mm256_div_ps(_mm256_mul_ps(num, x), den);
   return _mm256_max_ps(_mm256_set1_ps(-1.f), _mm256_min_ps(_mm256_set1_ps(1.f), num));
}

static OPUS_INLINE __m256 sigmoid_avx2(__m256 x)
{
   const __m256 half = _mm256_set1_ps(.5f);
   return _mm256_add_ps(half, _mm256_mul_ps(half, tansig_avx2(_mm256_mul_ps(half, x))));
}

/* out[i] += weights[j*col_stride+i]*x[j], j ascending, 8 outputs at a time. */
static OPUS_INLINE __m256 gemm8(__m256 acc, const opus_int8 *w, int cols, int col_stride, const float *x)
{
   int j;
   for (j=0;j<cols;j++)
   {
      __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w + j*col_stride))));
      acc = _mm256_add_ps(acc, _mm256_mul_ps(wf, _mm256_broadcast_ss(&x[j])));
   }
   return acc;
}

static OPUS_INLINE __m256 load_bias(const opus_int8 *b)
{
   return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)b)));
}

void analysis_compute_dense_avx2(const AnalysisDenseLayer *layer, float *output, const float *input)
{
   int i;
   int N = layer->nb_neurons, M = layer->nb_inputs;
   const __m256 scale = _mm256_set1_ps(WEIGHTS_SCALE);
   for (i=0;i<N;i+=8)
   {
      __m256 acc = gemm8(load_bias(&layer->bias[i]), &layer->input_weights[i], M, N, input);
      acc = _mm256_mul_ps(acc, scale);
      acc = layer->sigmoid ? sigmoid_avx2(acc) : tansig_avx2(acc);
      _mm256_storeu_ps(&output[i], acc);
   }
}

void analysis_compute_gru_avx2(const AnalysisGRULayer *gru, float *state, const float *input)
{
   int i;
   int N = gru->nb_neurons, M = gru->nb_inputs;
   int stride = 3*N;
   float rs[MAX_NEURONS];
   float z[MAX_NEURONS];
   const __m256 scale = _mm256_set1_ps(WEIGHTS_SCALE);
   const __m256 one = _mm256_set1_ps(1.f);
   for (i=0;i<N;i+=8)
   {
      __m256 a = gemm8(load_bias(&gru->bias[i]), &gru->input_weights[i], M, stride, input);
      a = gemm8(a, &gru->recurrent_weights[i], N, stride, state);
      _mm256_storeu_ps(&z[i], sigmoid_avx2(_mm256_mul_ps(scale, a)));
      a = gemm8(load_bias(&gru->bias[N+i]), &gru->input_weights[N+i], M, stride, input);
      a = gemm8(a, &gru->recurrent_weights[N+i], N, stride, state);
      __m256 r = sigmoid_avx2(_mm256_mul_ps(scale, a));
      _mm256_storeu_ps(&rs[i], _mm256_mul_ps(_mm256_loadu_ps(&state[i]), r));
   }
   /* All of tmp = state*r must exist before the third gemm. */
   {
      float h[MAX_NEURONS];
      for (i=0;i<N;i+=8)
      {
         __m256 a = gemm8(load_bias(&gru->bias[2*N+i]), &gru->input_weights[2*N+i], M, stride, input);
         a = gemm8(a, &gru->recurrent_weights[2*N+i], N, stride, rs);
         __m256 zz = _mm256_loadu_ps(&z[i]);
         __m256 t = tansig_avx2(_mm256_mul_ps(scale, a));
         __m256 hv = _mm256_add_ps(_mm256_mul_ps(zz, _mm256_loadu_ps(&state[i])),
                                   _mm256_mul_ps(_mm256_sub_ps(one, zz), t));
         _mm256_storeu_ps(&h[i], hv);
      }
      for (i=0;i<N;i++)
         state[i] = h[i];
   }
}

#endif
