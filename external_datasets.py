class RespTRAudioDataset(Dataset):
    def __init__(self, csv_path: str, binary: bool = True):
        self.df = pd.read_csv(csv_path)
        self.binary = binary

    def __len__(self):
        return len(self.df)


class COPDGoldTabularDataset(Dataset):
    def __init__(self, csv_path: str, feature_cols, label_col: str = "label_severe"):
        self.df = pd.read_csv(csv_path)
        self.feature_cols = feature_cols
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.feature_cols].values.astype(np.float32))
        y = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        return x, y
