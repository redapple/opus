# Bit accounting (exp/bitstats)

Build: `cmake -DCMAKE_C_FLAGS=-DOPUS_BITSTATS -DOPUS_BUILD_PROGRAMS=ON ..`
Run:   `opus_demo -e audio 48000 2 64000 -cbr -complexity 10 -framesize 20 in.pcm out.bit`
       `OPUS_BITSTATS=1 opus_demo -d 48000 2 out.bit out.pcm`
Counted on the decoder side (main CELT stream only; QEXT payload ignored).
`synth.py` makes a synthetic 30 s stereo test signal (placeholder until real clips exist).
