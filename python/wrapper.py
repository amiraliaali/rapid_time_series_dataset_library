import numpy as np
import numpy.typing as npt
import torch
from rust_time_series import ForecastingDataSet, ClassificationDataSet


class RustForecastingDataSet(ForecastingDataSet):
    def __new__(
        cls,
        data: npt.NDArray[np.float64] | torch.Tensor,
        train_prop: float,
        val_prop: float,
        test_prop: float,
    ):
        if isinstance(data, torch.Tensor):
            data = transform_to_numpy(data)

        return super().__new__(cls, data, train_prop, val_prop, test_prop)


class RustClassificationDataSet(ClassificationDataSet):
    def __new__(
        cls,
        data: npt.NDArray[np.float64] | torch.Tensor,
        labels: npt.NDArray[np.float64] | torch.Tensor,
        train_prop: float,
        val_prop: float,
        test_prop: float,
    ):
        if isinstance(data, torch.Tensor):
            data = transform_to_numpy(data)
        if isinstance(labels, torch.Tensor):
            labels = transform_to_numpy(labels)

        return super().__new__(cls, data, labels, train_prop, val_prop, test_prop)


def transform_to_numpy(
    data: npt.NDArray[np.float64] | torch.Tensor,
) -> npt.NDArray[np.float64]:
    return data.detach().cpu().numpy()
