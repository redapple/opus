/* Single source of truth for when the float x86 AVX2 kernels in celt/x86/
   (and their call sites in celt/ and src/) are compiled in.

   Include this after config.h. Do not repeat these conditions elsewhere:
   a mismatch between a kernel's guard and its call site shows up as a
   link error only in some build configurations.

   OPUS_CELT_AVX2          float build with runtime AVX2 dispatch available.
   OPUS_CELT_AVX2_NO_QEXT  ... and QEXT disabled (bands, maxabs16).
   OPUS_CELT_AVX2_MDCT     ... and QEXT and PFA disabled (MDCT pre/post
                           rotation and TDAC, which assume the radix FFT
                           MDCT layout).
   The radix butterflies (kiss_fft_avx2) are only reachable when
   opus_fft_impl() is compiled, and are guarded by OPUS_CELT_AVX2 alone;
   kiss_fft_use_avx2() must exist whenever OPUS_CELT_AVX2 is set. */
#ifndef CELT_X86_AVX2_GUARDS_H
#define CELT_X86_AVX2_GUARDS_H

#if !defined(FIXED_POINT) && defined(OPUS_X86_MAY_HAVE_AVX2)
# define OPUS_CELT_AVX2 1
# if !defined(ENABLE_QEXT)
#  define OPUS_CELT_AVX2_NO_QEXT 1
#  if !defined(ENABLE_PFA)
#   define OPUS_CELT_AVX2_MDCT 1
#  endif
# endif
#endif

#endif /* CELT_X86_AVX2_GUARDS_H */
