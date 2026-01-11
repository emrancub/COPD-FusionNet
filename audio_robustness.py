def evaluate_model_robustness(model, loader, model_type, noise_std):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            if "Audio" in model_type:
                inputs, labels = batch
                inputs = inputs.to(config.DEVICE)
                inputs = add_noise_to_spectrogram(inputs, noise_std)
                logits = model(inputs)
            elif "Fusion" in model_type:
                audio_x, tab_x, labels = batch
                audio_x = audio_x.to(config.DEVICE)
                tab_x = tab_x.to(config.DEVICE)

                audio_x = add_noise_to_spectrogram(audio_x, noise_std)
                logits = model(audio_x, tab_x)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())

    metrics = compute_metrics(all_labels, all_preds)
    return metrics['auc']

def run_robustness_analysis():
    print("[Robustness] Starting Noise Stress Test...")

    audio_df = pd.read_csv(config.AUDIO_METADATA_CSV)
    val_df = audio_df[audio_df['fold'] == 0]

    tab_c = pd.read_csv(config.TABULAR_COPD_CSV)
    tab_nc = pd.read_csv(config.TABULAR_NON_COPD_CSV)
    with open(config.TABULAR_FEATURES_JSON, 'r') as f: feats = json.load(f)

    ds_audio = COPDDataset(val_df, tab_c, tab_nc, feats, mode='audio', augment=False)

    ds_fusion = COPDDataset(val_df, tab_c, tab_nc, feats, mode='fusion', augment=False)

    loader_audio = DataLoader(ds_audio, batch_size=config.BATCH_SIZE, shuffle=False)
    loader_fusion = DataLoader(ds_fusion, batch_size=config.BATCH_SIZE, shuffle=False)

    model_eff = AudioModel_EffNet(pretrained=False).to(config.DEVICE)
    try:
        model_eff.load_state_dict(torch.load(os.path.join(config.CHECKPOINTS_DIR, "Audio_EffNet_fold0.pth")))
    except: print("Warning: Could not load Audio_EffNet weights.")

    model_crnn = AudioModel_CRNN().to(config.DEVICE)
    try:
        model_crnn.load_state_dict(torch.load(os.path.join(config.CHECKPOINTS_DIR, "Audio_CRNN_fold0.pth")))
    except: print("Warning: Could not load Audio_CRNN weights.")

    model_fusion = FusionNet_Quad(num_tab_features=len(feats)).to(config.DEVICE)
    try:
        model_fusion.load_state_dict(torch.load(os.path.join(config.CHECKPOINTS_DIR, "FusionNet_fold0.pth")))
    except: print("Warning: Could not load FusionNet weights.")


    print(f"\n{'Noise (std)':<12} | {'EffNet':<8} | {'CRNN':<8} | {'FusionNet':<8}")
    print("-" * 46)

    for noise in NOISE_LEVELS:
        auc_e = evaluate_model_robustness(model_eff, loader_audio, 'Audio_EffNet', noise)
        auc_c = evaluate_model_robustness(model_crnn, loader_audio, 'Audio_CRNN', noise)
        auc_f = evaluate_model_robustness(model_fusion, loader_fusion, 'FusionNet', noise)

        res_eff.append(auc_e)
        res_crnn.append(auc_c)
        res_fusion.append(auc_f)

        print(f"{noise:<12} | {auc_e:.4f}   | {auc_c:.4f}   | {auc_f:.4f}")

if __name__ == "__main__":
    run_robustness_analysis()