# reference: https://github.com/philipdarke/torchtime
from wrapper import RustClassificationDataSet, RustForecastingDataSet
import torch
from rust_time_series import (
    ImputeStrategy,
)


class RustPhysioNet2012(RustForecastingDataSet):
    def __new__(cls, split="train", train_prop: float = 0.7,
                val_prop: float = 0.15, test_prop: float = 0.15):
        try:
            from torchtime.data import PhysioNet2012
        except ImportError:
            raise ImportError("torchtime is not installed. Please `pip install torchtime`.")

        base_ds = PhysioNet2012(split=split, train_prop=train_prop)
        
        data = torch.as_tensor(base_ds.X, dtype=torch.float64)

        return super().__new__(
            cls,
            data=data.numpy(),
            train_prop=train_prop,
            val_prop=val_prop,
            test_prop=test_prop,
        )

class RustPhysioNet2019(RustClassificationDataSet):
    def __new__(cls, split="train", train_prop: float = 0.7,
                val_prop: float = 0.15, test_prop: float = 0.15):
        try:
            from torchtime.data import PhysioNet2019
        except ImportError:
            raise ImportError("torchtime is not installed. Please `pip install torchtime`.")

        base_ds = PhysioNet2019(split=split, train_prop=train_prop)
        
        data = torch.as_tensor(base_ds.X, dtype=torch.float64)
        labels = torch.as_tensor(base_ds.y, dtype=torch.float64)

        return super().__new__(
            cls,
            data=data.numpy(),
            labels=labels.numpy(),
            train_prop=train_prop,
            val_prop=val_prop,
            test_prop=test_prop,
        )
    
class RustUCRUEA(RustClassificationDataSet):
    def __new__(cls, dataset: str, train_prop: float = 0.7,
                val_prop: float = 0.15, test_prop: float = 0.15):
        try:
            from torchtime.data import UEA
        except ImportError:
            raise ImportError("torchtime is not installed. Please `pip install torchtime`.")

        base_ds = UEA(dataset=dataset, split="train", train_prop=train_prop)
        
        data = torch.as_tensor(base_ds.X, dtype=torch.float64)
        labels = torch.as_tensor(base_ds.y, dtype=torch.float64)

        return super().__new__(
            cls,
            data=data.numpy(),
            labels=labels.numpy(),
            train_prop=train_prop,
            val_prop=val_prop,
            test_prop=test_prop,
        )
    

def use_case():
    ds = RustPhysioNet2012()
    print(ds)
    ds.impute(ImputeStrategy.LeaveNaN)

    ds2 = RustPhysioNet2019()
    print(ds2)

    # list of datasets: https://timeseriesclassification.com/dataset.php
    ds3 = RustUCRUEA(dataset="ArrowHead", train_prop=0.7)
    print(ds3)

if __name__ == "__main__":
    use_case()
