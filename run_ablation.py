class Fusion_NoCRNN(nn.Module)
    def __init__(self, num_tab_features):
        super().__init__()
        self.branch_audio_eff = AudioModel_EffNet(pretrained=True)
        # CRNN Removed
        self.branch_tab_mlp = TabularModel_MLP(input_dim=num_tab_features)
        self.branch_tab_trans = TabularModel_Transformer(input_dim=num_tab_features)

        self.fusion_head = nn.Sequential(
            #We will make all necessary updates once the paper has been accepted.
        )

    def forward(self, audio_x, tab_x):
        _, e_a1 = self.branch_audio_eff(audio_x, return_embedding=True)
        # e_a2 (CRNN) skipped
        _, e_t1 = self.branch_tab_mlp(tab_x, return_embedding=True)
        _, e_t2 = self.branch_tab_trans(tab_x, return_embedding=True)

        concat = torch.cat([e_a1, e_t1, e_t2], dim=1)
        return self.fusion_head(concat)

class Fusion_NoTransformer(nn.Module):
    """Ablation: Removes Tabular Transformer Branch."""
    def __init__(self, num_tab_features):
        super().__init__()
        self.branch_audio_eff = AudioModel_EffNet(pretrained=True)
        self.branch_audio_crnn = AudioModel_CRNN()
        self.branch_tab_mlp = TabularModel_MLP(input_dim=num_tab_features)

        fusion_in = config.EMBED_DIM * 3

        self.fusion_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fusion_in, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, audio_x, tab_x):
        _, e_a1 = self.branch_audio_eff(audio_x, return_embedding=True)
        _, e_a2 = self.branch_audio_crnn(audio_x, return_embedding=True)
        _, e_t1 = self.branch_tab_mlp(tab_x, return_embedding=True)
        # e_t2 (Transformer) skipped

        concat = torch.cat([e_a1, e_a2, e_t1], dim=1)
        return self.fusion_head(concat)


def main():
    all_results = {}
    path_full = config.get_metrics_path("FusionNet")
    if os.path.exists(path_full):
        print("Loading existing Proposed FusionNet results...")
        all_results["Proposed_FusionNet"] = pd.read_csv(path_full)['auc'].values
    else:
        all_results["Proposed_FusionNet"] = train_ablation_model(None, "Proposed_FusionNet", use_aug=True)

    all_results["w/o Augmentation"] = train_ablation_model(FusionNet_Quad, "NoAug", use_aug=False)

    all_results["w/o Audio-CRNN"] = train_ablation_model(Fusion_NoCRNN, "NoCRNN", use_aug=True)

    all_results["w/o Tab-Transformer"] = train_ablation_model(Fusion_NoTransformer, "NoTrans", use_aug=True)

    plot_ablation_results(all_results)

if __name__ == "__main__":
    main()