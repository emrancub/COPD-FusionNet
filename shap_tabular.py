def load_data():
    print("[SHAP] Loading Data...")
    tab_c = pd.read_csv(config.TABULAR_COPD_CSV)
    tab_nc = pd.read_csv(config.TABULAR_NON_COPD_CSV)

    with open(config.TABULAR_FEATURES_JSON, 'r') as f:
        feature_names = json.load(f)

    df_c = tab_c[feature_names].sample(n=50, random_state=42)
    df_nc = tab_nc[feature_names].sample(n=50, random_state=42)
    background_df = pd.concat([df_c, df_nc])

    X_background = torch.tensor(background_df.values, dtype=torch.float32).to(config.DEVICE)

    return X_background, feature_names, background_df

def analyze_model(model_name, model_class, X_bg, feature_names, df_bg):
    print(f"\n--- Analyzing {model_name} ---")

    model = model_class(input_dim=len(feature_names)).to(config.DEVICE)
    weight_path = os.path.join(config.CHECKPOINTS_DIR, f"{model_name}_fold0.pth")

    if not os.path.exists(weight_path):
        print(f"Warning: Weights for {model_name} not found at {weight_path}. Skipping.")
        return

    try:
        model.load_state_dict(torch.load(weight_path, map_location=config.DEVICE))
    except:
        model.load_state_dict(torch.load(weight_path, map_location=config.DEVICE, weights_only=False))

    model.eval()

    def predict_fn(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(config.DEVICE)
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy().flatten()

    print(f"[SHAP] Computing values for {model_name}...")
    explainer = shap.KernelExplainer(predict_fn, shap.sample(df_bg, 50))
    shap_values = explainer.shap_values(df_bg, nsamples=100)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(14, 12))

    shap.summary_plot(
        shap_values,
        df_bg,
        plot_type="bar",
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )

    ax = plt.gca()
    ax.set_title(f"Feature Importance ({model_name})", fontweight='bold', fontsize=18, pad=20)
    ax.set_xlabel(
        "mean(|SHAP value|)\n(average impact on model output magnitude)",
        fontsize=14,
        fontweight='bold',
        labelpad=15,
    )

    fig.subplots_adjust(left=0.35, bottom=0.22, right=0.98, top=0.92)

    save_path_bar = os.path.join(config.PLOTS_DIR, f"Fig8_{model_name}_SHAP_Bar.png")
    fig.savefig(save_path_bar, dpi=config.DPI)
    print(f"Saved: {save_path_bar}")
    plt.close(fig)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(14, 12))

    shap.summary_plot(
        shap_values,
        df_bg,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )

    ax = plt.gca()
    ax.set_title(f"Feature Impact Direction ({model_name})", fontweight='bold', fontsize=18, pad=20)
    ax.set_xlabel(
        "SHAP value\n(impact on model output)",
        fontsize=14,
        fontweight='bold',
        labelpad=15,
    )

    fig.subplots_adjust(left=0.35, bottom=0.22, right=0.98, top=0.92)

    save_path_bee = os.path.join(config.PLOTS_DIR, f"Fig9_{model_name}_SHAP_Beeswarm.png")
    fig.savefig(save_path_bee, dpi=config.DPI)
    print(f"Saved: {save_path_bee}")
    plt.close(fig)


def main():
    X_bg, feats, df_bg = load_data()
    analyze_model("Tab_MLP", TabularModel_MLP, X_bg, feats, df_bg)
    analyze_model("Tab_Trans", TabularModel_Transformer, X_bg, feats, df_bg)
    print(f"\n[Done] SHAP figures saved to {config.PLOTS_DIR}")

if __name__ == "__main__":
    main()