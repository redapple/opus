/* AVX2 version of celt_maxabs16().

   Same licence as the rest of Opus (BSD-3-clause, see COPYING).

   Uses the same compare-and-select as the scalar MAX16/MIN16 (so NaN
   samples are ignored exactly like the scalar code), and max/min are exact,
   so the result is identical. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if !defined(FIXED_POINT) && !defined(ENABLE_QEXT) && defined(OPUS_X86_MAY_HAVE_AVX2)

#include <immintrin.h>
#include "opus_types.h"
#include "opus_defines.h"
#include "arch.h"

float celt_maxabs16_avx2(const float *x, int len)
{
   int i = 0;
   __m256 vmax = _mm256_setzero_ps();
   __m256 vmin = _mm256_setzero_ps();
   float maxval = 0, minval = 0;
   float tmp[8];
   for (; i+8 <= len; i += 8)
   {
      __m256 v = _mm256_loadu_ps(&x[i]);
      vmax = _mm256_blendv_ps(vmax, v, _mm256_cmp_ps(v, vmax, _CMP_GT_OQ));
      vmin = _mm256_blendv_ps(vmin, v, _mm256_cmp_ps(v, vmin, _CMP_LT_OQ));
   }
   _mm256_storeu_ps(tmp, vmax);
   for (int k=0;k<8;k++) maxval = tmp[k] > maxval ? tmp[k] : maxval;
   _mm256_storeu_ps(tmp, vmin);
   for (int k=0;k<8;k++) minval = tmp[k] < minval ? tmp[k] : minval;
   for (; i < len; i++)
   {
      maxval = x[i] > maxval ? x[i] : maxval;
      minval = x[i] < minval ? x[i] : minval;
   }
   return maxval > -minval ? maxval : -minval;
}

#endif
