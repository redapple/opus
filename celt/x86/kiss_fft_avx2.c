/* Copyright (c) 2026 contributors
   AVX butterflies for the float kiss_fft.

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

/* Design notes
   ------------
   These kernels vectorise the radix-2/3/4/5 butterflies across the inner
   (j/u) loop, four complex values per __m256, interleaved as r,i,r,i,...
   Every lane performs exactly the same IEEE single-precision operations,
   in the same order, as the scalar C macros in _kiss_fft_guts.h:
     - complex multiply = two separate muls then add/sub (never FMA),
     - x + (-y) is used in place of x - y only where that is exact,
     - commuted additions (a*b + c*d vs c*d + a*b) are exact.
   The result is bit-identical to the C path, so encoder output does not
   change. The file is built with -mavx2 -mfma, so contraction of
   mul+add into FMA must be disabled explicitly below. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if !defined(FIXED_POINT) && defined(OPUS_X86_MAY_HAVE_AVX2)

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

#include <immintrin.h>
#include <stddef.h>
#include "_kiss_fft_guts.h"
#include "x86/kiss_fft_avx2.h"

#define SWAP_RI(x) _mm256_permute_ps((x), 0xB1)

static OPUS_INLINE __m256 sign_odd(void)
{
   return _mm256_castsi256_ps(_mm256_set1_epi64x((long long)0x8000000000000000ULL));
}

/* Load four twiddles tw[0], tw[s], tw[2s], tw[3s]. */
static OPUS_INLINE __m256 ld_tw(const kiss_twiddle_cpx *tw, size_t s)
{
   __m128d lo, hi;
   if (s == 1) return _mm256_loadu_ps((const float*)tw);
   lo = _mm_loadh_pd(_mm_load_sd((const double*)tw), (const double*)(tw + s));
   hi = _mm_loadh_pd(_mm_load_sd((const double*)(tw + 2*s)), (const double*)(tw + 3*s));
   return _mm256_castpd_ps(_mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1));
}

/* C_MUL: r = a.r*w.r - a.i*w.i ; i = a.r*w.i + a.i*w.r */
static OPUS_INLINE __m256 cmul(__m256 a, __m256 w)
{
   __m256 wr = _mm256_moveldup_ps(w);
   __m256 wi = _mm256_movehdup_ps(w);
   return _mm256_addsub_ps(_mm256_mul_ps(a, wr), _mm256_mul_ps(SWAP_RI(a), wi));
}

/* [x.r + y.i, x.i - y.r] */
static OPUS_INLINE __m256 rot_a(__m256 x, __m256 y)
{
   return _mm256_add_ps(x, _mm256_xor_ps(SWAP_RI(y), sign_odd()));
}

/* [x.r - y.i, x.i + y.r] */
static OPUS_INLINE __m256 rot_b(__m256 x, __m256 y)
{
   return _mm256_addsub_ps(x, SWAP_RI(y));
}

#define LD(p) _mm256_loadu_ps((const float*)(p))
#define ST(p, v) _mm256_storeu_ps((float*)(p), (v))

void kf_bfly2_avx2(kiss_fft_cpx *Fout, int N)
{
   int i;
   const __m256 tw = _mm256_set1_ps(0.7071067812f);
   const __m256 so = sign_odd();
   for (i = 0; i < N; i++)
   {
      __m256 a = LD(Fout);
      __m256 b = LD(Fout + 4);
      __m256 bs = SWAP_RI(b);
      __m256 p = _mm256_add_ps(b, bs);   /* [r+i, i+r] */
      __m256 d = _mm256_sub_ps(b, bs);   /* [r-i, i-r] */
      __m256 q = _mm256_sub_ps(bs, b);   /* [i-r, r-i] */
      __m256 M = _mm256_blend_ps(p, d, 0xAA);                     /* [r+i, i-r] */
      __m256 Nn = _mm256_blend_ps(q, _mm256_xor_ps(p, so), 0xAA); /* [i-r, -(i+r)] */
      __m256 t13 = _mm256_mul_ps(_mm256_blend_ps(M, Nn, 0xF0), tw);
      __m256 t02 = _mm256_blend_ps(b, _mm256_xor_ps(bs, so), 0xF0);
      __m256 t = _mm256_blend_ps(t02, t13, 0xCC);
      ST(Fout + 4, _mm256_sub_ps(a, t));
      ST(Fout, _mm256_add_ps(a, t));
      Fout += 8;
   }
}

void kf_bfly4_avx2(kiss_fft_cpx *Fout, const size_t fstride,
      const kiss_fft_state *st, int m, int N, int mm)
{
   int i;
   if (m == 1)
   {
      int N4 = N & ~3;
      for (i = 0; i < N4; i += 4)
      {
         __m256d v0 = _mm256_castps_pd(LD(Fout));
         __m256d v1 = _mm256_castps_pd(LD(Fout + 4));
         __m256d v2 = _mm256_castps_pd(LD(Fout + 8));
         __m256d v3 = _mm256_castps_pd(LD(Fout + 12));
         __m256d t0 = _mm256_unpacklo_pd(v0, v1);
         __m256d t1 = _mm256_unpackhi_pd(v0, v1);
         __m256d t2 = _mm256_unpacklo_pd(v2, v3);
         __m256d t3 = _mm256_unpackhi_pd(v2, v3);
         __m256 c0 = _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t2, 0x20));
         __m256 c2 = _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t2, 0x31));
         __m256 c1 = _mm256_castpd_ps(_mm256_permute2f128_pd(t1, t3, 0x20));
         __m256 c3 = _mm256_castpd_ps(_mm256_permute2f128_pd(t1, t3, 0x31));
         __m256 s0, s1, o0, o1, o2, o3;
         __m256d u0, u1, u2, u3;

         s0 = _mm256_sub_ps(c0, c2);
         c0 = _mm256_add_ps(c0, c2);
         s1 = _mm256_add_ps(c1, c3);
         o2 = _mm256_sub_ps(c0, s1);
         o0 = _mm256_add_ps(c0, s1);
         s1 = _mm256_sub_ps(c1, c3);
         o1 = rot_a(s0, s1);
         o3 = rot_b(s0, s1);

         u0 = _mm256_unpacklo_pd(_mm256_castps_pd(o0), _mm256_castps_pd(o1));
         u1 = _mm256_unpackhi_pd(_mm256_castps_pd(o0), _mm256_castps_pd(o1));
         u2 = _mm256_unpacklo_pd(_mm256_castps_pd(o2), _mm256_castps_pd(o3));
         u3 = _mm256_unpackhi_pd(_mm256_castps_pd(o2), _mm256_castps_pd(o3));
         ST(Fout,      _mm256_castpd_ps(_mm256_permute2f128_pd(u0, u2, 0x20)));
         ST(Fout + 4,  _mm256_castpd_ps(_mm256_permute2f128_pd(u1, u3, 0x20)));
         ST(Fout + 8,  _mm256_castpd_ps(_mm256_permute2f128_pd(u0, u2, 0x31)));
         ST(Fout + 12, _mm256_castpd_ps(_mm256_permute2f128_pd(u1, u3, 0x31)));
         Fout += 16;
      }
      /* Scalar tail, identical to the C reference. */
      for (; i < N; i++)
      {
         kiss_fft_cpx scratch0, scratch1;
         C_SUB(scratch0, *Fout, Fout[2]);
         C_ADDTO(*Fout, Fout[2]);
         C_ADD(scratch1, Fout[1], Fout[3]);
         C_SUB(Fout[2], *Fout, scratch1);
         C_ADDTO(*Fout, scratch1);
         C_SUB(scratch1, Fout[1], Fout[3]);
         Fout[1].r = scratch0.r + scratch1.i;
         Fout[1].i = scratch0.i - scratch1.r;
         Fout[3].r = scratch0.r - scratch1.i;
         Fout[3].i = scratch0.i + scratch1.r;
         Fout += 4;
      }
   } else {
      const int m2 = 2*m, m3 = 3*m;
      const kiss_twiddle_cpx *tw = st->twiddles;
      kiss_fft_cpx *Fout_beg = Fout;
      for (i = 0; i < N; i++)
      {
         int j;
         Fout = Fout_beg + i*mm;
         for (j = 0; j < m; j += 4)
         {
            __m256 a0 = LD(Fout + j);
            __m256 s0 = cmul(LD(Fout + m + j),  ld_tw(tw + j*fstride,   fstride));
            __m256 s1 = cmul(LD(Fout + m2 + j), ld_tw(tw + 2*j*fstride, 2*fstride));
            __m256 s2 = cmul(LD(Fout + m3 + j), ld_tw(tw + 3*j*fstride, 3*fstride));
            __m256 s5 = _mm256_sub_ps(a0, s1);
            __m256 s3, s4;
            a0 = _mm256_add_ps(a0, s1);
            s3 = _mm256_add_ps(s0, s2);
            s4 = _mm256_sub_ps(s0, s2);
            ST(Fout + m2 + j, _mm256_sub_ps(a0, s3));
            ST(Fout + j, _mm256_add_ps(a0, s3));
            ST(Fout + m + j, rot_a(s5, s4));
            ST(Fout + m3 + j, rot_b(s5, s4));
         }
      }
   }
}

void kf_bfly3_avx2(kiss_fft_cpx *Fout, const size_t fstride,
      const kiss_fft_state *st, int m, int N, int mm)
{
   int i;
   const int m2 = 2*m;
   const kiss_twiddle_cpx *tw = st->twiddles;
   const __m256 epi3i = _mm256_set1_ps(st->twiddles[fstride*m].i);
   const __m256 half = _mm256_set1_ps(.5f);
   kiss_fft_cpx *Fout_beg = Fout;
   for (i = 0; i < N; i++)
   {
      int j;
      Fout = Fout_beg + i*mm;
      for (j = 0; j < m; j += 4)
      {
         __m256 a0 = LD(Fout + j);
         __m256 s1 = cmul(LD(Fout + m + j),  ld_tw(tw + j*fstride,   fstride));
         __m256 s2 = cmul(LD(Fout + m2 + j), ld_tw(tw + 2*j*fstride, 2*fstride));
         __m256 s3 = _mm256_add_ps(s1, s2);
         __m256 s0 = _mm256_sub_ps(s1, s2);
         __m256 fm = _mm256_sub_ps(a0, _mm256_mul_ps(s3, half));
         s0 = _mm256_mul_ps(s0, epi3i);
         ST(Fout + j, _mm256_add_ps(a0, s3));
         ST(Fout + m2 + j, rot_a(fm, s0));
         ST(Fout + m + j, rot_b(fm, s0));
      }
   }
}

void kf_bfly5_avx2(kiss_fft_cpx *Fout, const size_t fstride,
      const kiss_fft_state *st, int m, int N, int mm)
{
   int i;
   const kiss_twiddle_cpx *tw = st->twiddles;
   const kiss_twiddle_cpx ya = st->twiddles[fstride*m];
   const kiss_twiddle_cpx yb = st->twiddles[fstride*2*m];
   const __m256 yar = _mm256_set1_ps(ya.r), yai = _mm256_set1_ps(ya.i);
   const __m256 ybr = _mm256_set1_ps(yb.r), ybi = _mm256_set1_ps(yb.i);
   const __m256 so = sign_odd();
   kiss_fft_cpx *Fout_beg = Fout;
   for (i = 0; i < N; i++)
   {
      int u;
      kiss_fft_cpx *F0 = Fout_beg + i*mm;
      kiss_fft_cpx *F1 = F0 + m, *F2 = F0 + 2*m, *F3 = F0 + 3*m, *F4 = F0 + 4*m;
      for (u = 0; u < m; u += 4)
      {
         __m256 s0 = LD(F0 + u);
         __m256 s1 = cmul(LD(F1 + u), ld_tw(tw + u*fstride,   fstride));
         __m256 s2 = cmul(LD(F2 + u), ld_tw(tw + 2*u*fstride, 2*fstride));
         __m256 s3 = cmul(LD(F3 + u), ld_tw(tw + 3*u*fstride, 3*fstride));
         __m256 s4 = cmul(LD(F4 + u), ld_tw(tw + 4*u*fstride, 4*fstride));
         __m256 s7 = _mm256_add_ps(s1, s4);
         __m256 s10 = _mm256_sub_ps(s1, s4);
         __m256 s8 = _mm256_add_ps(s2, s3);
         __m256 s9 = _mm256_sub_ps(s2, s3);
         __m256 s5, s6, s11, s12, t, uu, vv;

         ST(F0 + u, _mm256_add_ps(s0, _mm256_add_ps(s7, s8)));

         s5 = _mm256_add_ps(s0, _mm256_add_ps(_mm256_mul_ps(s7, yar), _mm256_mul_ps(s8, ybr)));
         t = _mm256_add_ps(_mm256_mul_ps(s10, yai), _mm256_mul_ps(s9, ybi));
         s6 = _mm256_xor_ps(SWAP_RI(t), so);               /* [t.i, -t.r] */
         ST(F1 + u, _mm256_sub_ps(s5, s6));
         ST(F4 + u, _mm256_add_ps(s5, s6));

         s11 = _mm256_add_ps(s0, _mm256_add_ps(_mm256_mul_ps(s7, ybr), _mm256_mul_ps(s8, yar)));
         uu = _mm256_sub_ps(_mm256_mul_ps(s9, yai), _mm256_mul_ps(s10, ybi));
         vv = _mm256_sub_ps(_mm256_mul_ps(s10, ybi), _mm256_mul_ps(s9, yai));
         s12 = _mm256_blend_ps(SWAP_RI(uu), SWAP_RI(vv), 0xAA); /* [uu.i, vv.r] */
         ST(F2 + u, _mm256_add_ps(s11, s12));
         ST(F3 + u, _mm256_sub_ps(s11, s12));
      }
   }
}

#endif
