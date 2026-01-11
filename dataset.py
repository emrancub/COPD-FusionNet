class COPDDataset(Dataset):
    def __init__(self, audio_df, tab_copd, tab_non_copd, tab_features, mode='fusion', augment=False):
        self.audio_df = audio_df.reset_index(drop=True)
        self.tab_copd = tab_copd
        self.tab_non_copd = tab_non_copd
        self.tab_features = tab_features
        self.mode = mode
        self.augment = augment

        self.augmenter = StrongAudioAugmentations(p=0.6)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
