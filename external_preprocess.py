def build_external_audio_meta() -> None:
    labels = pd.read_excel(config.EXTERNAL_AUDIO_LABELS_XLSX)

    if "Patient ID" not in labels.columns or "Diagnosis" not in labels.columns:
        raise ValueError(
            f"Labels.xlsx must contain columns 'Patient ID' and 'Diagnosis', "
            f"but found: {list(labels.columns)}"
        )

    labels["Patient ID"] = labels["Patient ID"].astype(str).str.strip()
    labels["Diagnosis"] = labels["Diagnosis"].astype(str).str.strip().str.upper()
    pid_to_diag = dict(zip(labels["Patient ID"], labels["Diagnosis"]))

    rows = []

    if not os.path.isdir(config.EXTERNAL_AUDIO_DIR):
        raise FileNotFoundError(
            f"Audio directory not found: {config.EXTERNAL_AUDIO_DIR}"
        )

    for fname in os.listdir(config.EXTERNAL_AUDIO_DIR):
        if not fname.lower().endswith(".wav"):
            continue

        patient_id = fname.split("_")[0]

        diag = pid_to_diag.get(patient_id)
        if diag is None:
            continue

        if not diag.startswith("COPD"):
            continue

        try:
            stage = int(diag.replace("COPD", ""))
        except ValueError:
            continue

        copd_stage = stage

        label_binary = 0 if copd_stage == 0 else 1

        rows.append(
            {
                "file_path": os.path.join(config.EXTERNAL_AUDIO_DIR, fname),
                "patient_id": patient_id,
                "diag_str": diag,
                "copd_stage": copd_stage,
                "label_binary": label_binary,
            }
        )

    os.makedirs(os.path.dirname(config.EXTERNAL_AUDIO_META_CSV), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.EXTERNAL_AUDIO_META_CSV, index=False)
def main():
    build_external_audio_meta()


if __name__ == "__main__":
    main()
