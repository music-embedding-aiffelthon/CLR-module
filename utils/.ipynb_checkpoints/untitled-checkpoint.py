def freq_mask(x):

    v = x.shape[1] # no. of mel bins

    # apply m_F frequency masks to the mel spectrogram
    for i in range(self.m_F):
        self.rng, key = jax.random.split(self.rng)
        f = int(jax.random.uniform(key, shape=(1,), minval=0, maxval=self.F)) # [0, F)
        f0 = jax.random.randint(key, shape=(1,), minval=0, maxval= v - f) # [0, v - f)
        print(f)
        print(f0)
        x = x.at[:, f0[0]:f0[0]+ f, :, :].set(0)

    tau = x.shape[2] # time frames

    # apply m_T time masks to the mel spectrogram
    for i in range(self.m_T):
        self.rng, key = jax.random.split(self.rng)
        t = int(jax.random.uniform(key,  shape=(1,), minval=0, maxval=self.T)) # [0, T)
        t0 = jax.random.randint(key, shape=(1,), minval=0, maxval=tau - t) # [0, tau - t)
        print(t)
        print(t0)
        x = x.at[:, :, t0[0]:t0[0] + t, :].set(0)

    
    return x


def time_mask(x):

    tau = x.shape[2] # time frames

    # apply m_T time masks to the mel spectrogram
    for i in range(self.m_T):
        self.rng, key = jax.random.split(self.rng)
        t = int(jax.random.uniform(key,  shape=(1,), minval=0, maxval=self.T)) # [0, T)
        t0 = jax.random.randint(key, shape=(1,), minval=0, maxval=tau - t) # [0, tau - t)
        print(t)
        print(t0)
        x = x.at[:, :, t0[0]:t0[0] + t, :].set(0)

    return x