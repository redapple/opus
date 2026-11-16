/* Experimental decoder-side bit accounting (exp/bitstats branch).
   Enabled with -DOPUS_BITSTATS; print with env OPUS_BITSTATS=1.
   Counts the main CELT stream only (QEXT extension payload ignored).
   Decoder-side so that encoder trial encodes (theta_rdo) don't double count. */
#ifndef BITSTATS_H
#define BITSTATS_H
#ifdef OPUS_BITSTATS
#include "entcode.h"
enum { BS_HEADER, BS_COARSE, BS_TF, BS_SIDE, BS_ALLOC, BS_FINE, BS_BANDS, BS_FINAL, BS_NCAT };
typedef struct {
   long long cat[BS_NCAT];     /* 1/8 bit units */
   long long pulses;           /* PVQ index bits (inside BS_BANDS) */
   long long theta;            /* theta/split/inv bits (inside BS_BANDS) */
   long long band_pulses[32];
   long long band_theta[32];
   long long frames, samples48, transients, silent;
   long long lm_hist[4];
   opus_int32 last;
   int registered;
} bitstats_t;
extern bitstats_t opus_bitstats;
void opus_bitstats_register(void);
#define BS_START(dec) do{ opus_bitstats_register(); opus_bitstats.last=ec_tell_frac(dec);}while(0)
#define BS_MARK(dec,c) do{ opus_int32 _t=ec_tell_frac(dec); opus_bitstats.cat[c]+=_t-opus_bitstats.last; opus_bitstats.last=_t;}while(0)
#else
#define BS_START(dec) do{}while(0)
#define BS_MARK(dec,c) do{}while(0)
#endif
#endif
