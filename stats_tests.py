def load_metrics(model_name):
    path = config.get_metrics_path(model_name)
    try:
        df = pd.read_csv(path)
        return df.sort_values('fold')['auc'].values
    except FileNotFoundError:
        print(f"Warning: Metrics file not found for {model_name} at {path}")
        return None

def perform_tests():
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TESTS (Paired t-test on AUC)")
    print("="*60)
    print(f"{'Comparison':<35} | {'Delta (Mean)':<12} | {'p-value':<10} | {'Sig.'}")
    print("-" * 75)

    fusion_auc = load_metrics('FusionNet')

    if fusion_auc is None:
        print("Error: FusionNet metrics missing. Run train.py first.")
        return

    for baseline_name in MODELS_TO_COMPARE:
        baseline_auc = load_metrics(baseline_name)

        if baseline_auc is not None:
            if len(fusion_auc) != len(baseline_auc):
                print(f"Error: Fold count mismatch for {baseline_name}")
                continue

            t_stat, p_val = ttest_rel(fusion_auc, baseline_auc)


            mean_diff = fusion_auc.mean() - baseline_auc.mean()

            # Significance stars
            if p_val < 0.001: sig = "***"
            elif p_val < 0.01: sig = "**"
            elif p_val < 0.05: sig = "*"
            else: sig = "ns"

            print(f"FusionNet vs. {baseline_name:<20} | {mean_diff:+.4f}       | {p_val:.4f}     | {sig}")

    print("-" * 75)
    print("* p<0.05, ** p<0.01, *** p<0.001")

if __name__ == "__main__":
    perform_tests()