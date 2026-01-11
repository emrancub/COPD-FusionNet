#We will make all necessary updates once the paper has been accepted.
PATH_DS1_ROOT =
PATH_DS2_ROOT =
PATH_DS3_CSV  =

def load_ds1_metadata(root_path):
    print("[DS1] Loading ICBHI metadata...")
    try:
        diag_path = os.path.join(root_path, "patient_diagnosis.csv")
        diag_df = pd.read_csv(diag_path, names=['patient_id', 'diagnosis'])

        demo_path = os.path.join(root_path, "demographic_info.txt")
        demo_df = pd.read_csv(demo_path, sep=r'\s+', names=['patient_id', 'age', 'sex', 'adult_bmi', 'child_weight', 'child_height'])

        df = pd.merge(diag_df, demo_df, on='patient_id', how='left')
        df['label'] = df['diagnosis'].apply(lambda x: 1 if x == 'COPD' else 0)
        return df
    except Exception as e:
        print(f"[Error] Could not load DS1 metadata: {e}")
        return pd.DataFrame()

def process_audio_files(ds1_root, ds2_root, ds1_meta):
    audio_data = []

    ds1_audio_dir = os.path.join(ds1_root, "audio_and_txt_files")
    if os.path.exists(ds1_audio_dir):
        print("[Audio] Scanning DS1 files...")
        for fname in os.listdir(ds1_audio_dir):
            if fname.endswith('.wav'):
                try:
                    pid = int(fname.split('_')[0])
                    patient_row = ds1_meta[ds1_meta['patient_id'] == pid]
                    if not patient_row.empty:
                        label = patient_row.iloc[0]['label']
                        audio_data.append({
                            'patient_id': pid,
                            'file_path': os.path.join(ds1_audio_dir, fname),
                            'label': label,
                            'source': 'DS1'
                        })
                except:
                    continue
    else:
        print(f"[Warning] DS1 Audio folder not found: {ds1_audio_dir}")

    ds2_audio_dir = os.path.join(ds2_root, "Audio Files")
    if os.path.exists(ds2_audio_dir):
        print("[Audio] Scanning DS2 files...")
        pattern = re.compile(r"^[BDE]P(\d+)_([a-zA-Z0-9\s+]+),.*\.wav$")

        for fname in os.listdir(ds2_audio_dir):
            if fname.endswith('.wav'):
                match = pattern.match(fname)
                if match:
                    pid = 2000 + int(match.group(1))
                    condition = match.group(2).lower()
                    label = 1 if 'copd' in condition else 0

                    audio_data.append({
                        'patient_id': pid,
                        'file_path': os.path.join(ds2_audio_dir, fname),
                        'label': label,
                        'source': 'DS2'
                    })
    else:
        print(f"[Warning] DS2 Audio folder not found: {ds2_audio_dir}")

    return pd.DataFrame(audio_data)

def create_folds(df):
    print("[Folds] Generating 5-fold split...")

    patient_labels = df.groupby('patient_id')['label'].max()

    patient_fold_map = {}
    patients = patient_labels.index.values
    labels = patient_labels.values

    for fold, (train_idx, val_idx) in enumerate(skf.split(patients, labels)):
        val_patients = patients[val_idx]
        for pid in val_patients:
            patient_fold_map[pid] = fold

    df['fold'] = df['patient_id'].map(patient_fold_map)
    return df

def process_tabular(ds3_path):
    print("[Tabular] Processing DS3...")
    if not os.path.exists(ds3_path):
        print(f"[Error] Tabular CSV not found: {ds3_path}")
        return

def main():

if __name__ == "__main__":
    main()