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

#include "x86/mdct_avx2.h"

#ifdef MDCT_HAVE_AVX2

#include "x86/float_avx2.h"

#define DUP_EVEN(x) _mm256_moveldup_ps(x)
#define DUP_ODD(x) _mm256_movehdup_ps(x)

static OPUS_INLINE __m256 reverse8(__m256 x)
{
   return _mm256_permutevar8x32_ps(x, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
}

/* Both post-rotations write the pair (a_k, b_k) computed for complex index k
   to out[2*k] and out[N2-1-2*k]. Given f = (a,b) for k = i..i+3 and g = (a,b)
   for the mirrored k = j..j+3 (j = N4-4-i, so that N2-1-2*k for one block
   falls in the other one), this does those writes as two contiguous
   stores: out[2*i..] = (a_i, b_{j+3}, a_{i+1}, b_{j+2}, ...), and the
   other way round for out[2*j..]. */
static OPUS_INLINE void store_mirrored(float *out, int i, int j, __m256 f, __m256 g)
{
   /* Only the odd lanes matter: b_{k+3}, b_{k+2}, b_{k+1}, b_k. */
   const __m256i rev_b = _mm256_setr_epi32(0, 7, 0, 5, 0, 3, 0, 1);
   _mm256_storeu_ps(out + 2*i, _mm256_blend_ps(f, _mm256_permutevar8x32_ps(g, rev_b), 0xAA));
   _mm256_storeu_ps(out + 2*j, _mm256_blend_ps(g, _mm256_permutevar8x32_ps(f, rev_b), 0xAA));
}

/* Forward MDCT pre-rotation, with the scaling and bit-reversal. */
void clt_mdct_prerot_avx2(const float *f, const float *t, float scale,
      const opus_int16 *bitrev, kiss_fft_cpx *f2, int N4)
{
   int i;
   const __m256 sc = _mm256_set1_ps(scale);
   for (i=0;i+4<=N4;i+=4)
   {
      __m256 x = _mm256_loadu_ps(f + 2*i);
      __m256 w = _mm256_loadu_ps(t + 2*i);
      /* (re*t1 - im*t0, im*t1 + re*t0) */
      __m256 y = cpx_rot_avx2(x, DUP_ODD(w), DUP_EVEN(w));
      cpx_scatter4_avx2(f2, bitrev + i, _mm256_mul_ps(y, sc));
   }
   for (;i<N4;i++)
   {
      float re = f[2*i], im = f[2*i+1], t0 = t[2*i], t1 = t[2*i+1];
      float yr = re*t1 - im*t0;
      float yi = im*t1 + re*t0;
      f2[bitrev[i]].r = yr*scale;
      f2[bitrev[i]].i = yi*scale;
   }
}

/* Forward MDCT post-rotation, stride 1:
   out[2*i] = yr_i, out[N2-1-2*i] = yi_i, with N2 = 2*N4. */
void clt_mdct_postrot_avx2(const kiss_fft_cpx *fp, const float *t,
      float *out, int N4)
{
   int i, j;
   for (i=0;i+4<=N4-4-i;i+=4)
   {
      __m256 f, g, wf, wg;
      j = N4-4-i;
      f = _mm256_loadu_ps((const float*)(fp + i));
      g = _mm256_loadu_ps((const float*)(fp + j));
      wf = _mm256_loadu_ps(t + 2*i);
      wg = _mm256_loadu_ps(t + 2*j);
      /* (yr, yi) = (fi*t0 - fr*t1, fr*t0 + fi*t1) */
      f = cpx_rot_avx2(CPX_SWAP_AVX2(f), DUP_EVEN(wf), DUP_ODD(wf));
      g = cpx_rot_avx2(CPX_SWAP_AVX2(g), DUP_EVEN(wg), DUP_ODD(wg));
      store_mirrored(out, i, j, f, g);
   }
   /* Middle part, not covered by a pair of blocks. */
   for (j=N4-i;i<j;i++)
   {
      float fr = fp[i].r, fi = fp[i].i, t0 = t[2*i], t1 = t[2*i+1];
      out[2*i] = fi*t0 - fr*t1;
      out[2*N4-1-2*i] = fr*t0 + fi*t1;
   }
}

/* Backward MDCT pre-rotation, stride 1. With x1 = in[2*i] and
   x2 = in[N2-1-2*i]: y[bitrev[i]] = (x1*t1 - x2*t0, x2*t1 + x1*t0). */
void clt_imdct_prerot_avx2(const float *in, const float *t,
      const opus_int16 *bitrev, kiss_fft_cpx *y, int N4)
{
   int i;
   const int N2 = 2*N4;
   for (i=0;i+4<=N4;i+=4)
   {
      /* x1 and x2 for each i, both duplicated to (x, x). x2 is read from
         the end of the buffer, so it comes in reverse order. */
      __m256 x1 = DUP_EVEN(_mm256_loadu_ps(in + 2*i));
      __m256 x2 = reverse8(DUP_ODD(_mm256_loadu_ps(in + N2-8-2*i)));
      __m256 w = _mm256_loadu_ps(t + 2*i);
      /* (x1, x2)*t1 -+ (x2, x1)*t0 */
      __m256 r = _mm256_addsub_ps(
            _mm256_mul_ps(_mm256_blend_ps(x1, x2, 0xAA), DUP_ODD(w)),
            _mm256_mul_ps(_mm256_blend_ps(x2, x1, 0xAA), DUP_EVEN(w)));
      cpx_scatter4_avx2(y, bitrev + i, r);
   }
   for (;i<N4;i++)
   {
      float x1 = in[2*i], x2 = in[N2-1-2*i];
      float t0 = t[2*i], t1 = t[2*i+1];
      y[bitrev[i]].r = x1*t1 - x2*t0;
      y[bitrev[i]].i = x2*t1 + x1*t0;
   }
}

/* Backward MDCT post-rotation and de-shuffle, in place on y (2*N4 floats).
   Element k is (re, im) = (y[2*k+1], y[2*k]) and gives
   y[2*k] = re*t1 + im*t0 and y[N2-1-2*k] = re*t0 - im*t1. Elements k and
   N4-1-k write each other's inputs, so they are always processed together. */
void clt_imdct_postrot_avx2(float *y, const float *t, int N4)
{
   int i, j;
   for (i=0;i+4<=N4-4-i;i+=4)
   {
      __m256 f, g, wf, wg;
      j = N4-4-i;
      f = _mm256_loadu_ps(y + 2*i);
      g = _mm256_loadu_ps(y + 2*j);
      wf = _mm256_loadu_ps(t + 2*i);
      wg = _mm256_loadu_ps(t + 2*j);
      /* Even lanes: re*t1 + im*t0, odd lanes: re*t0 - im*t1. */
      {
         __m256 a = _mm256_mul_ps(CPX_SWAP_AVX2(f), DUP_ODD(wf));
         __m256 b = _mm256_mul_ps(f, DUP_EVEN(wf));
         f = _mm256_blend_ps(_mm256_add_ps(a, b), _mm256_sub_ps(b, a), 0xAA);
         a = _mm256_mul_ps(CPX_SWAP_AVX2(g), DUP_ODD(wg));
         b = _mm256_mul_ps(g, DUP_EVEN(wg));
         g = _mm256_blend_ps(_mm256_add_ps(a, b), _mm256_sub_ps(b, a), 0xAA);
      }
      store_mirrored(y, i, j, f, g);
   }
   /* Middle part. Same as the C loop: when N4 is odd, the middle element is
      computed twice. */
   for (;i<(N4+1)>>1;i++)
   {
      float re, im, t0, t1, yr, yi;
      j = N4-1-i;
      re = y[2*i+1];
      im = y[2*i];
      t0 = t[2*i];
      t1 = t[2*i+1];
      yr = re*t1 + im*t0;
      yi = re*t0 - im*t1;
      re = y[2*j+1];
      im = y[2*j];
      y[2*i] = yr;
      y[2*j+1] = yi;
      t0 = t[2*j];
      t1 = t[2*j+1];
      yr = re*t1 + im*t0;
      yi = re*t0 - im*t1;
      y[2*j] = yr;
      y[2*i+1] = yi;
   }
}

/* TDAC mirror on the first overlap samples: with x1 = out[overlap-1-i],
   x2 = out[i], w1 = window[i] and w2 = window[overlap-1-i]:
   out[i] = x2*w2 - x1*w1 and out[overlap-1-i] = x2*w1 + x1*w2. */
void clt_imdct_tdac_avx2(float *out, const float *window, int overlap)
{
   int i, j;
   for (i=0;i+8<=overlap-8-i;i+=8)
   {
      __m256 x1, x2, w1, w2;
      j = overlap-8-i;
      x2 = _mm256_loadu_ps(out + i);
      x1 = reverse8(_mm256_loadu_ps(out + j));
      w1 = _mm256_loadu_ps(window + i);
      w2 = reverse8(_mm256_loadu_ps(window + j));
      _mm256_storeu_ps(out + i, _mm256_sub_ps(_mm256_mul_ps(x2, w2), _mm256_mul_ps(x1, w1)));
      _mm256_storeu_ps(out + j, reverse8(_mm256_add_ps(_mm256_mul_ps(x2, w1), _mm256_mul_ps(x1, w2))));
   }
   for (;i<overlap/2;i++)
   {
      float x1 = out[overlap-1-i], x2 = out[i];
      float w1 = window[i], w2 = window[overlap-1-i];
      out[i] = x2*w2 - x1*w1;
      out[overlap-1-i] = x2*w1 + x1*w2;
   }
}

#endif
