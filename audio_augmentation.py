class StrongAudioAugmentations(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

        self.freq_mask = T.FrequencyMasking(freq_mask_param=20)
        self.time_mask = T.TimeMasking(time_mask_param=60)

    def add_noise(self, waveform):
        if torch.rand(1) < self.p:
            noise_level = np.random.uniform(0.001, 0.015)
            noise = torch.randn_like(waveform) * noise_level
            return waveform + noise
        return waveform

    def time_shift(self, waveform):
        if torch.rand(1) < self.p:
            shift_amt = int(np.random.uniform(-0.1, 0.1) * waveform.shape[-1])
            return torch.roll(waveform, shift_amt, dims=-1)
        return waveform