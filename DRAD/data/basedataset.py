from typing import Any, Dict

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        X,
        y,
    ):
        super(BaseDataset, self).__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"x": self.X[index], "y": self.y[index]}

    def __len__(self):
        return len(self.X)
