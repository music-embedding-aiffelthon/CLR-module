# https://github.com/pyyush/SpecAugment/blob/219fc6e9ed4838fe9700295700040b1da283c536/augment.py

def spec_augment(x, policy):
    
    if policy == 'LB':
        W, F, m_F, T, p, m_T = 48, 6, 1, 300, 1.0, 1
    elif policy == 'LD':
        W, F, m_F, T, p, m_T = 48, 6, 2, 300, 1.0, 2

    v = x.shape[1] # no. of mel bins

    # apply m_F frequency masks to the mel spectrogram
    for i in range(m_F):
        rng, key = jax.random.split(rng)
        f = int(jax.random.uniform(key, shape=(1,), minval=0, maxval=F)) # [0, F)
        f0 = jax.random.randint(key, shape=(1,), minval=0, maxval= v - f) # [0, v - f)
        print(f)
        print(f0)
        x = x.at[:, f0[0]:f0[0]+ f, :, :].set(0)

    tau = x.shape[2] # time frames

    # apply m_T time masks to the mel spectrogram
    for i in range(m_T):
        rng, key = jax.random.split(rng)
        t = int(jax.random.uniform(key,  shape=(1,), minval=0, maxval=T)) # [0, T)
        t0 = jax.random.randint(key, shape=(1,), minval=0, maxval=tau - t) # [0, tau - t)
        print(t)
        print(t0)
        x = x.at[:, :, t0[0]:t0[0] + t, :].set(0)

    return x


