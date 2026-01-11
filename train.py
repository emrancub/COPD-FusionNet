def load_data():
    audio_df = pd.read_csv(config.AUDIO_METADATA_CSV)
    tab_c = pd.read_csv(config.TABULAR_COPD_CSV)
    tab_nc = pd.read_csv(config.TABULAR_NON_COPD_CSV)
    with open(config.TABULAR_FEATURES_JSON, 'r') as f:
        feats = json.load(f)
    return audio_df, tab_c, tab_nc, feats

def get_model_instance(model_name, num_tab_feats):
    if model_name == "Audio_EffNet": return AudioModel_EffNet(pretrained=True)
    if model_name == "Audio_CRNN":   return AudioModel_CRNN()
    if model_name == "Tab_MLP":      return TabularModel_MLP(input_dim=num_tab_feats)
    if model_name == "Tab_Trans":    return TabularModel_Transformer(input_dim=num_tab_feats)
    if model_name == "FusionNet":    return FusionNet_Quad(num_tab_features=num_tab_feats)
    raise ValueError(f"Unknown Model: {model_name}")

def train_one_epoch(model, loader, criterion, optimizer, device, mode):
    model.train()
    running_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()

        if mode == "audio":
            x, y = batch
            x, y = x.to(device), y.to(device).unsqueeze(1)
            logits = model(x)
        elif mode == "tab":
            x, y = batch
            x, y = x.to(device), y.to(device).unsqueeze(1)
            logits = model(x)
        else:
            xa, xt, y = batch
            xa, xt, y = xa.to(device), xt.to(device), y.to(device).unsqueeze(1)
            logits = model(xa, xt)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device, mode):)

def run_experiment(model_name):

if __name__ == "__main__":
    # Sequential training
    run_experiment("Audio_EffNet")
    run_experiment("Audio_CRNN")
    run_experiment("Tab_MLP")
    run_experiment("Tab_Trans")
    run_experiment("FusionNet")