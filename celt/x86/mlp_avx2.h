#ifndef MLP_AVX2_H
#define MLP_AVX2_H

#include "../../src/mlp.h"

#if !defined(FIXED_POINT) && defined(OPUS_X86_MAY_HAVE_AVX2)
#define MLP_HAVE_AVX2 1
/* Require layer->nb_neurons % 8 == 0. Bit-identical to the C versions. */
void analysis_compute_dense_avx2(const AnalysisDenseLayer *layer, float *output, const float *input);
void analysis_compute_gru_avx2(const AnalysisGRULayer *gru, float *state, const float *input);
#endif

#endif
