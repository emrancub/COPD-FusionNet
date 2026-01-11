def main():
    print("Starting evaluation process...")

    print("Loading test dataset...")
    test_dataset = COPDDataset(
        audio_meta_csv=config.PREPROCESSED_AUDIO_META,
        tabular_copd_csv=config.PREPROCESSED_TABULAR_COPD,
        tabular_non_copd_csv=config.PREPROCESSED_TABULAR_NON_COPD,
        mode='test'
    )

    if len(test_dataset) == 0:
        print("Error: No test samples found. Did you run 'old_preprocess.py'?")
        return

    num_tabular_features = test_dataset.tabular_feature_count
    print(f"Using {len(test_dataset)} samples for testing.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

if __name__ == "__main__":
    main()

