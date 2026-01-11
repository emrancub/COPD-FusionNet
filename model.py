# ==========================================
# 1. Audio Branch A: EfficientNet-B2
# ==========================================
class AudioModel_EffNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b2',
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            drop_rate=0.3,
            drop_path_rate=0.2
        )
        num_features = self.backbone.num_features

        self.projection = nn.Sequential(
            #We will make all necessary updates once the paper has been accepted.
        )
        self.classifier = nn.Linear(config.EMBED_DIM, 1)

    def forward(self, x, return_embedding=False):
        feat = self.backbone(x)
        emb = self.projection(feat)
        logits = self.classifier(emb)
        return (logits, emb) if return_embedding else logits

# ==========================================
# 2. Audio Branch B: CRNN with Attention
# ==========================================
class AudioModel_CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        in_ch = 1
        for out_ch in config.CRNN_CNN_CHANNELS:
            #We will make all necessary updates once the paper has been accepted.
        self.cnn = nn.Sequential(*layers)

        self.rnn_input_dim = config.CRNN_CNN_CHANNELS[-1] * 8

        self.rnn = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=config.CRNN_RNN_HIDDEN,
            num_layers=config.CRNN_RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Self-Attention pooling layer
        self.attention_fc = nn.Linear(config.CRNN_RNN_HIDDEN * 2, 1)

        self.projection = nn.Sequential(
            nn.Linear(config.CRNN_RNN_HIDDEN * 2, config.EMBED_DIM),
            nn.GELU()
        )
        self.classifier = nn.Linear(config.EMBED_DIM, 1)

    def forward(self, x, return_embedding=False):
        x = self.cnn(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, -1)

        x, _ = self.rnn(x)

        att_scores = self.attention_fc(x) # [B, T, 1]
        att_weights = torch.softmax(att_scores, dim=1)

        x = torch.sum(x * att_weights, dim=1)

        emb = self.projection(x)
        logits = self.classifier(emb)
        return (logits, emb) if return_embedding else logits

# ==========================================
# 3. Tabular Branch A: ResMLP
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class TabularModel_MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        #We will make all necessary updates once the paper has been accepted.
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(256, config.EMBED_DIM),
            nn.GELU()
        )
        self.classifier = nn.Linear(config.EMBED_DIM, 1)

    def forward(self, x, return_embedding=False):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        emb = self.output_proj(x)
        logits = self.classifier(emb)
        return (logits, emb) if return_embedding else logits

# ==========================================
# 4. Tabular Branch B: Transformer
# ==========================================
class TabularModel_Transformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.num_tokens = 8
        self.token_dim = 64

        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, self.num_tokens * self.token_dim),
            nn.GELU()
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, self.token_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            #We will make all necessary updates once the paper has been accepted.
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.TRANSFORMER_LAYERS)

        self.projection = nn.Sequential(
            nn.Linear(self.token_dim * self.num_tokens, config.EMBED_DIM),
            nn.GELU()
        )
        self.classifier = nn.Linear(config.EMBED_DIM, 1)

# ==========================================
# 5. COPD-FusionNet: Gated Quad-Fusion
# ==========================================
class FusionNet_Quad(nn.Module):
    def __init__(self, num_tab_features):
        super().__init__()
        self.branch_audio_eff = AudioModel_EffNet(pretrained=True)
        self.branch_audio_crnn = AudioModel_CRNN()
        self.branch_tab_mlp = TabularModel_MLP(input_dim=num_tab_features)
        self.branch_tab_trans = TabularModel_Transformer(input_dim=num_tab_features)

        fusion_in = config.EMBED_DIM * 4

        self.gate = nn.Sequential(
            nn.Linear(fusion_in, 4),
            nn.Softmax(dim=1)
        )

        self.fusion_head = nn.Sequential(
            #We will make all necessary updates once the paper has been accepted.
        )