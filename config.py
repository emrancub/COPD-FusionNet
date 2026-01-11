# ---------------------------------------------------------
# 1. System & Device
# ---------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"[System] Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("[System] WARNING: Using CPU. Training will be slow.")

# ---------------------------------------------------------
# 2. Project Paths
# ---------------------------------------------------------
PROJECT_OUTPUT_DIR = "project_output"
RESULTS_DIR = os.path.join(PROJECT_OUTPUT_DIR, "results")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PLOTS_DIR = os.path.join(PROJECT_OUTPUT_DIR, "plots")

os.makedirs(PROJECT_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Processed Data Paths
AUDIO_METADATA_CSV = os.path.join(PROJECT_OUTPUT_DIR, "audio_metadata_5fold.csv")
TABULAR_COPD_CSV   = os.path.join(PROJECT_OUTPUT_DIR, "tabular_copd_processed.csv")
TABULAR_NON_COPD_CSV = os.path.join(PROJECT_OUTPUT_DIR, "tabular_non_copd_processed.csv")
TABULAR_FEATURES_JSON = os.path.join(PROJECT_OUTPUT_DIR, "tabular_features_list.json")

# ---------------------------------------------------------
# 3. Hyperparameters
# ---------------------------------------------------------
NUM_FOLDS = 5
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 1e-3
PATIENCE = 15

# ---------------------------------------------------------
# 4. Audio Parameters
# ---------------------------------------------------------
SAMPLE_RATE = 22050
DURATION = 5.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
F_MIN = 50
F_MAX = 11000

# ---------------------------------------------------------
# 5. Model Architecture Configs
# ---------------------------------------------------------
EMBED_DIM = 128

CRNN_CNN_CHANNELS = [32, 64, 128, 256] # Deeper CNN
CRNN_RNN_HIDDEN = 128
CRNN_RNN_LAYERS = 2

TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 3
TRANSFORMER_DIM_FF = 256

# ---------------------------------------------------------
# 6. Plotting Standards
# ---------------------------------------------------------
DPI = 1000
FONT_FAMILY = 'serif'
FONT_SIZE = 14
FONT_WEIGHT = 'bold'

def get_metrics_path(model_name):
    return os.path.join(RESULTS_DIR, f"metrics_{model_name}.csv")

def get_pred_path(model_name, fold):
    return os.path.join(PREDICTIONS_DIR, f"preds_{model_name}_fold_{fold}.npz")