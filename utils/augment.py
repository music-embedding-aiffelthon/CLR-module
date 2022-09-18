# https://github.com/pyyush/SpecAugment/blob/219fc6e9ed4838fe9700295700040b1da283c536/augment.py
import jax


class SpecAugment():

    
    def __init__(self, mel_spectrogram, policy, rng):
        self.mel_spectrogram = mel_spectrogram
        self.policy = policy
        self.rng = rng
        # Policy Specific Parameters
        if self.policy == 'LB':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 48, 6, 1, 300, 1.0, 1
        elif self.policy == 'LD':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 48, 6, 2, 300, 1.0, 2
        
    
    def freq_mask(self):
        
        v = self.mel_spectrogram.shape[1] # no. of mel bins
        
        # apply m_F frequency masks to the mel spectrogram
        for i in range(self.m_F):
            self.rng, key = jax.random.split(self.rng)
            f = int(jax.random.uniform(key, shape=(1,), minval=0, maxval=self.F)) # [0, F)
            f0 = jax.random.randint(key, shape=(1,), minval=0, maxval= v - f) # [0, v - f)
            print(f)
            print(f0)
            self.mel_spectrogram = self.mel_spectrogram.at[:, f0[0]:f0[0]+ f, :, :].set(0)
            
        for i in range(self.m_T):
            self.rng, key = jax.random.split(self.rng)
            t = int(jax.random.uniform(key,  shape=(1,), minval=0, maxval=self.T)) # [0, T)
            t0 = jax.random.randint(key, shape=(1,), minval=0, maxval=tau - t) # [0, tau - t)
            print(t)
            print(t0)
            self.mel_spectrogram = self.mel_spectrogram.at[:, :, t0[0]:t0[0] + t, :].set(0)
        return self.mel_spectrogram
    
    
    def time_mask(self):
    
        tau = self.mel_spectrogram.shape[2] # time frames
        
        # apply m_T time masks to the mel spectrogram
        for i in range(self.m_T):
            self.rng, key = jax.random.split(self.rng)
            t = int(jax.random.uniform(key,  shape=(1,), minval=0, maxval=self.T)) # [0, T)
            t0 = jax.random.randint(key, shape=(1,), minval=0, maxval=tau - t) # [0, tau - t)
            print(t)
            print(t0)
            self.mel_spectrogram = self.mel_spectrogram.at[:, :, t0[0]:t0[0] + t, :].set(0)
            
        return self.mel_spectrogram