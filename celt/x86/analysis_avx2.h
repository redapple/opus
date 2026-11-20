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

#ifndef ANALYSIS_AVX2_H
#define ANALYSIS_AVX2_H

#include "kiss_fft.h"
#include "x86cpu.h"
#include "../../src/mlp.h"

#if defined(OPUS_X86_MAY_HAVE_AVX2) && !defined(FIXED_POINT)
#define ANALYSIS_HAVE_AVX2

/* AVX2 versions of analysis_compute_dense() and analysis_compute_gru(),
   vectorised over the neurons. Bit-exact with the C code. nb_neurons must
   be a multiple of 8 (and at most MAX_NEURONS for the GRU). */
void analysis_compute_dense_avx2(const AnalysisDenseLayer *layer, float *output, const float *input);
void analysis_compute_gru_avx2(const AnalysisGRULayer *gru, float *state, const float *input);

/* Per-bin phase and tonality loop of tonality_analysis(), for the bins
   1..N2-1 of the N = 2*N2 point FFT output. Processes the bins by blocks of
   8 and returns the first bin left to the C loop. Bit-exact with it.
   tonality_scale is the 40.f*16.f*pi4 factor of the C code. */
int tonality_bins_avx2(const kiss_fft_cpx *out, int N2, float *A, float *dA,
      float *d2A, float *tonality, float *tonality2, float *noisiness,
      float tonality_scale);
#endif

#endif
