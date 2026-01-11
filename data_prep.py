def process_internal_data():
    print("Processing Internal Training Data...")
    audio_files = []

    try:
        diag = pd.read_csv(os.path.join(config.PATH_DS1, "patient_diagnosis.csv"), names=['pid', 'diag'])
        adir = os.path.join(config.PATH_DS1, "audio_and_txt_files")
        for f in os.listdir(adir):
            if f.endswith('.wav'):
                pid = int(f.split('_')[0])
                d = diag[diag['pid'] == pid]['diag'].values[0]
                audio_files.append({'path': os.path.join(adir, f), 'pid': f"k_{pid}", 'label': map_diagnosis_binary(d)})
    except Exception as e:
        print(f"DS1 Error: {e}")

    try:
        adir = os.path.join(config.PATH_DS2, "Audio Files")
        for f in os.listdir(adir):
            if f.endswith('.wav'):
                # Filename: BP108_COPD,E W,P R L ,63,M.wav
                parts = f.split(',')
                if len(parts) > 1:
                    diag_part = parts[0].split('_')[1]
                    audio_files.append({'path': os.path.join(adir, f), 'pid': f"m_{f.split('_')[0]}",
                                        'label': map_diagnosis_binary(diag_part)})
    except Exception as e:
        print(f"DS2 Error: {e}")

    audio_df = pd.DataFrame(audio_files)

    audio_df['fold'] = -1
    p_labels = audio_df.groupby('pid')['label'].max()  # If patient has COPD sample, they are COPD
    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED)
    for fold, (_, val_idx) in enumerate(skf.split(p_labels.index, p_labels.values)):
        val_pids = p_labels.index[val_idx]
        audio_df.loc[audio_df['pid'].isin(val_pids), 'fold'] = fold

    audio_df.to_csv(config.TRAIN_AUDIO_CSV, index=False)

    try:
        df = pd.read_csv(config.PATH_DS3)
        df.columns = [c.lower().strip() for c in df.columns]
        df['label'] = df['disease'].apply(map_diagnosis_binary)


        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(50)
        df['male'] = (df['sex'].str.lower() == 'male').astype(int)

        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        syms = ohe.fit_transform(df[['symptoms']].astype(str))
        feat_names = ['age', 'male'] + list(ohe.get_feature_names_out())
        scaler = StandardScaler()
        df['age'] = scaler.fit_transform(df[['age']])

        final_tab = pd.concat([df[['label', 'age', 'male']], pd.DataFrame(syms, columns=ohe.get_feature_names_out())],
                              axis=1)
        final_tab.to_csv(config.TRAIN_TAB_CSV, index=False)

        with open(config.TAB_FEATURES_JSON, 'w') as f:
            json.dump(feat_names, f)
    except Exception as e:
        print(f"DS3 Error: {e}")


def process_external_data():
    print("Processing External Validation Data...")

    try:
        lbl = pd.read_excel(os.path.join(config.PATH_EXT_AUDIO_ROOT, "Labels.xlsx"))
        adir = os.path.join(config.PATH_EXT_AUDIO_ROOT, "audio")
        ext_audio = []
        for f in os.listdir(adir):
            if f.endswith('.wav'):
                pid = f.split('.')[0].split('_')[0]  # H002_L1 -> H002
                row = lbl[lbl['Patient ID'] == pid]
                if not row.empty:
                    diag = str(row.iloc[0]['Diagnosis']).upper()
                    # Strict Mapping
                    label = 0 if diag == 'COPD0' else (1 if 'COPD' in diag else -1)
                    if label != -1:
                        ext_audio.append({'path': os.path.join(adir, f), 'label': label})
        pd.DataFrame(ext_audio).to_csv(config.EXT_AUDIO_CSV, index=False)
        print(f"  Ext Audio: {len(ext_audio)} samples.")
    except Exception as e:
        print(f"Ext Audio Error: {e}")

    try:
        # Try CSV first if user converted it, else xlsx
        path_csv = os.path.join(config.PATH_EXT_TAB_ROOT, "PatientCategorical.csv")
        if os.path.exists(path_csv):
            df = pd.read_csv(path_csv)
        else:
            df = pd.read_excel(os.path.join(config.PATH_EXT_TAB_ROOT, "230PatientsCOPD.xlsx"))

        target_col = [c for c in df.columns if 'GOLD' in c.upper()][0]
        df['label'] = df[target_col].apply(lambda x: 1 if x in [1, 2, 3, 4] else 0)  # Assuming 0 is healthy if present

        df.to_csv(config.EXT_TAB_CSV, index=False)
        print(f"  Ext Tabular: {len(df)} samples.")
    except Exception as e:
        print(f"Ext Tabular Error: {e}")


if __name__ == "__main__":
    ensure_dirs()
    process_internal_data()
    process_external_data()