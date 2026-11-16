import numpy as np, sys
sr=48000; T=30; n=sr*T; rng=np.random.default_rng(1)
t=np.arange(n)/sr
L=np.zeros(n); R=np.zeros(n)
# chord progression of harmonic tones with decay envelopes
notes=[220,277.2,329.6,440,554.4,659.3,110,146.8]
for k in range(60):
    t0=k*0.5; f=notes[k%len(notes)]*(2.0**int(rng.integers(-1,2)))
    m=(t>=t0)&(t<t0+2.5); tt=t[m]-t0
    env=np.exp(-tt*1.5)*(1-np.exp(-tt*80))
    sig=sum(env*np.sin(2*np.pi*f*h*tt+rng.uniform(0,6))/h**1.2 for h in range(1,14) if f*h<20000)
    pan=rng.uniform(0.2,0.8); L[m]+=sig*pan; R[m]+=sig*(1-pan)
# percussive noise bursts (hats/snare-like)
for k in range(int(T*4)):
    t0=k*0.25; m=(t>=t0)&(t<t0+0.12); tt=t[m]-t0
    nz=rng.standard_normal(m.sum())*np.exp(-tt*40)*(0.4 if k%2 else 0.15)
    L[m]+=nz; R[m]+=nz*0.9
# low pad + slow noise
pad=np.sin(2*np.pi*55*t)*0.2; L+=pad; R+=pad
L+=rng.standard_normal(n)*0.002; R+=rng.standard_normal(n)*0.002
x=np.stack([L,R],1); x*=0.7/np.abs(x).max()
(x*32767).astype('<i2').tofile(sys.argv[1])
