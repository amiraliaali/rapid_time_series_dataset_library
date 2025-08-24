import numpy as np
import numpy.typing as npt
import torch
from rust_time_series import ForecastingDataSet, ClassificationDataSet
from typing import Union


class RustForecastingDataSet(ForecastingDataSet):
    def __new__(
        cls,
        data: Union[npt.NDArray[np.float64], torch.Tensor],
        train_prop: float,
        val_prop: float,
        test_prop: float,
    ):
        was_tensor = isinstance(data, torch.Tensor)
        if was_tensor:
            data = transform_to_numpy(data)

        self = super().__new__(cls, data, train_prop, val_prop, test_prop)

        self._was_tensor = was_tensor

        return self

    def collect(self, past_window: int, future_horizon: int, stride: int):
        res = super().collect(past_window, future_horizon, stride)

        if self._was_tensor:
            res = _to_tensor(res)

        return res


class RustClassificationDataSet(ClassificationDataSet):
    def __new__(
        cls,
        data: Union[npt.NDArray[np.float64], torch.Tensor],
        labels: Union[npt.NDArray[np.float64], torch.Tensor],
        train_prop: float,
        val_prop: float,
        test_prop: float,
    ):
        was_tensor = isinstance(data, torch.Tensor) and isinstance(labels, torch.Tensor)

        if not was_tensor and (
            isinstance(data, torch.Tensor) or isinstance(labels, torch.Tensor)
        ):
            raise TypeError(
                "Inconsistent tensor types: both data and labels must be tensors or numpy arrays."
            )

        if was_tensor:
            data = transform_to_numpy(data)
            labels = transform_to_numpy(labels)

        self = super().__new__(cls, data, labels, train_prop, val_prop, test_prop)

        self._was_tensor = was_tensor

        return self

    def collect(self):
        res = super().collect()

        if self._was_tensor:
            res = _to_tensor(res)

        return res


def transform_to_numpy(
    data: torch.Tensor,
) -> npt.NDArray[np.float64]:
    check_tensor_is_double(data)
    return data.detach().cpu().numpy()


def check_tensor_is_double(data: torch.Tensor):
    if data.dtype != torch.float64:
        raise TypeError(
            f"Tensor is not of type double (float64). Actual dtype: {data.dtype}. "
            "You can convert it using `data = data.to(torch.float64)`."
        )


def _to_tensor(obj):
    """
    Recursively convert numpy.ndarray to torch.Tensor.
    Leave other objects untouched.
    """
    if isinstance(obj, np.ndarray):
        tensor = torch.from_numpy(obj)
        return tensor
    elif isinstance(obj, tuple):
        return tuple(_to_tensor(item) for item in obj)
    elif isinstance(obj, list):
        return [_to_tensor(item) for item in obj]
    else:
        return obj
