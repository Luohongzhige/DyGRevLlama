from torch.utils.data import Dataset

class HistoryEmbeddingDataset(Dataset):
    
    def __init__(self):
        self.reviewer_history = []
        self.asin_history = []
        self.embedding = []
        self.text = []
    
    def append(self, reviewer_history, asin_history, embedding, text):
        self.reviewer_history.append(reviewer_history)
        self.asin_history.append(asin_history)
        self.embedding.append(embedding)
        self.text.append(text)
        
    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        return self.reviewer_history[idx], self.asin_history[idx], self.embedding[idx], self.text[idx]