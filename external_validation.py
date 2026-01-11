def calculate_metrics_raw(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = auc(roc_curve(y_true, y_pred_prob)[0], roc_curve(y_true, y_pred_prob)[1])
        else:
            roc_auc = 0.5
    except:
        roc_auc = 0.5

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)

    return {"auc": roc_auc, "accuracy": acc, "f1": f1, "sensitivity": sens, "specificity": spec, "mcc": mcc}

def bootstrap_confidence_intervals(y_true, y_pred_prob, n_bootstraps=BOOTSTRAP_ROUNDS):
    metrics_list = []
    for _ in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)), replace=True)
        if len(np.unique(np.array(y_true)[indices])) < 2: continue
        boot_y_true = np.array(y_true)[indices]
        boot_y_pred = np.array(y_pred_prob)[indices]
        metrics_list.append(calculate_metrics_raw(boot_y_true, boot_y_pred))

    df_boot = pd.DataFrame(metrics_list)
    results = {}
    for col in df_boot.columns:
        mean = df_boot[col].mean()
        std = df_boot[col].std()
        results[col] = f"{mean:.4f} ± {std:.4f}"
        results[f"{col}_mean"] = mean
    return results


def load_external_tabular(csv_path, train_feature_names):
    print(f"[External] Loading Tabular Data...")
    df = pd.read_csv(csv_path)
    df['label'] = df['disease_name'].apply(lambda x: 1 if str(x).strip().upper() == 'COPD' else 0)
    processed_df = pd.DataFrame(0, index=np.arange(len(df)), columns=train_feature_names)

    if 'age' in df.columns and 'age' in train_feature_names:
        processed_df['age'] = df['age'].fillna(df['age'].median())
    if 'sex' in df.columns:
        if 'is_male' in train_feature_names:
            processed_df['is_male'] = df['sex'].apply(lambda x: 1 if str(x).strip().lower() == 'male' else 0)
        if 'is_female' in train_feature_names:
            processed_df['is_female'] = df['sex'].apply(lambda x: 1 if str(x).strip().lower() == 'female' else 0)
    if 'symptoms' in df.columns:
        for i, row in df.iterrows():
            symptoms_str = str(row['symptoms']).lower()
            for feat in train_feature_names:
                if feat.startswith('symptoms_'):
                    symptom_core = feat.replace('symptoms_', '').replace('_', ' ')
                    if symptom_core in symptoms_str:
                        processed_df.loc[i, feat] = 1.0

    return processed_df.values.astype(np.float32), df['label'].values.astype(np.float32)

def load_external_audio(labels_path, audio_dir):
    print(f"[External] Loading Audio Data...")
    try:
        df = pd.read_excel(labels_path)
    except:
        csv_path = labels_path.replace(".xlsx", ".csv")
        if os.path.exists(csv_path):
             df = pd.read_csv(csv_path)
        else:
             csv_path_alt = os.path.join(os.path.dirname(labels_path), "Labels.xlsx - Sayfa1.csv")
             if os.path.exists(csv_path_alt):
                 df = pd.read_csv(csv_path_alt)
             else:
                 raise FileNotFoundError("Could not find Labels file.")

    audio_data = []
    for i, row in df.iterrows():
        pid = str(row['Patient ID'])
        diagnosis = str(row['Diagnosis']).strip().upper()
        label = 0 if diagnosis == 'COPD0' else 1

        found_file = None
        for fname in os.listdir(audio_dir):
            if fname.startswith(pid) and fname.endswith('.wav'):
                found_file = os.path.join(audio_dir, fname)
                break
        if found_file:
            audio_data.append({"file_path": found_file, "label": label})

    print(f"  Found {len(audio_data)} matching audio files.")
    return pd.DataFrame(audio_data)

class ExternalDataset(Dataset):
    def __init__(self, audio_df, tab_X, tab_y, mode='fusion', noise_level=0.0, missing_rate=0.0):
        self.audio_df = audio_df
        self.tab_X = tab_X
        self.tab_y = tab_y
        self.mode = mode
        self.noise_level = noise_level
        self.missing_rate = missing_rate
        self.neg_indices = np.where(self.tab_y == 0)[0]
        self.pos_indices = np.where(self.tab_y == 1)[0]
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=config.SAMPLE_RATE, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, n_mels=config.N_MELS)
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self): return len(self.audio_df)

    def _get_tabular(self, label):
        pool = self.pos_indices if label == 1 else self.neg_indices
        if len(pool) == 0: idx = np.random.randint(0, len(self.tab_X))
        else: idx = np.random.choice(pool)
        feat_vector = self.tab_X[idx].copy()

        # Heavy Stress Test: Randomly zero out 30% of features
        if self.missing_rate > 0:
            mask = np.random.rand(*feat_vector.shape) < self.missing_rate
            feat_vector[mask] = 0
        return torch.tensor(feat_vector, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.audio_df.iloc[idx]
        label = float(row['label'])
        try:
            wav, sr = torchaudio.load(row['file_path'])
            if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
            if sr != config.SAMPLE_RATE: wav = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)(wav)
            if wav.shape[1] < config.NUM_SAMPLES: wav = torch.nn.functional.pad(wav, (0, config.NUM_SAMPLES - wav.shape[1]))
            else: wav = wav[:, :config.NUM_SAMPLES]

            # Heavy Stress Test: Add 15% Noise
            if self.noise_level > 0:
                wav = wav + torch.randn_like(wav) * self.noise_level

            mel = self.db_transform(self.mel_transform(wav))
        except: mel = torch.zeros((1, config.N_MELS, 216))

        tab = self._get_tabular(label)

        if self.mode == 'audio': return mel, torch.tensor(label, dtype=torch.float32)
        elif self.mode == 'tab': return tab, torch.tensor(label, dtype=torch.float32)
        else: return mel, tab, torch.tensor(label, dtype=torch.float32)


def evaluate_model(model_name, model_class, loader, device, mode='fusion'):
    print(f"\nEvaluating {model_name}...")

    model = model_class().to(device) if mode == "audio" else \
            model_class(input_dim=len(loader.dataset.tab_X[0])).to(device) if mode == "tab" else \
            model_class(num_tab_features=len(loader.dataset.tab_X[0])).to(device)

    n_params = count_parameters(model)

    weight_path = os.path.join(config.CHECKPOINTS_DIR, f"{model_name}_fold0.pth")
    if not os.path.exists(weight_path):
        print(f"  Warning: Weights not found.")
        return None, None, None, n_params

    try:
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
    except: return None, None, None, n_params

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if mode == 'audio': logits = model(batch[0].to(device))
            elif mode == 'tab': logits = model(batch[0].to(device))
            else: logits = model(batch[0].to(device), batch[1].to(device))
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(batch[-1].numpy())

    metrics = bootstrap_confidence_intervals(all_labels, all_preds)
    return metrics, all_labels, all_preds, n_params


if __name__ == "__main__":
    evaluate_external_suite()