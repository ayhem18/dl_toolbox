from typing import Any, Dict

from .base import BaseLogger

_EXTRA = "tensorboard"


class TensorBoardLogger(BaseLogger):
    @staticmethod
    def _import_tensorboard():
        TensorBoardLogger._import_package("tensorboard", extra=_EXTRA)
        torch = TensorBoardLogger._import_package("torch", extra=_EXTRA)
        try:
            return torch.utils.tensorboard.SummaryWriter
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "torch.utils.tensorboard.SummaryWriter could not be imported. "
                "Install the tensorboard extra:\n"
                f"  uv add 'dl-toolbox[{_EXTRA}]'\n"
                f"  pip install 'dl-toolbox[{_EXTRA}]'"
            ) from exc

    @staticmethod
    def _import_torch():
        return TensorBoardLogger._import_package("torch", extra=_EXTRA)

    @staticmethod
    def _import_numpy():
        return TensorBoardLogger._import_package("numpy", extra=_EXTRA)

    @staticmethod
    def _import_pandas():
        return TensorBoardLogger._import_package("pandas", extra=_EXTRA)

    def __init__(self, log_dir: str, **kwargs):
        super().__init__(log_dir, **kwargs)
        SummaryWriter = self._import_tensorboard()
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalar(self, step: int, tag: str, value: float):
        self.writer.add_scalar(tag, value, global_step=step)

    def log_dict(self, step: int, values: Dict[str, float]):
        for tag, value in values.items():
            self.log_scalar(step, tag, value)

    def log_image(self, step: int, tag: str, image: Any, **kwargs):
        torch = self._import_torch()
        np = self._import_numpy()
        dataformats = kwargs.pop("dataformats", None)
        if dataformats is None:
            if isinstance(image, torch.Tensor):
                dataformats = "CHW"
            elif isinstance(image, np.ndarray):
                dataformats = "HWC"
            else:
                raise TypeError(
                    f"Image must be a torch.Tensor or np.ndarray, got {type(image)}"
                )
        self.writer.add_image(tag, image, step, dataformats=dataformats, **kwargs)

    def log_table(self, step: int, tag: str, data: Any, **kwargs):
        pd = self._import_pandas()
        try:
            if isinstance(data, pd.DataFrame):
                text_table = data.to_markdown()
            else:
                text_table = pd.DataFrame(data).to_markdown()
        except Exception as e:
            print(f"Error logging table to tensorboard: {e}")
            try:
                text_table = str(data)
            except TypeError:
                raise TypeError(
                    "The data can be converted to neither a pandas Dataframe, markdown nor a string"
                )
        self.writer.add_text(tag, text_table, global_step=step, **kwargs)

    def log_histogram(self, step: int, tag: str, values: Any):
        self.writer.add_histogram(tag, values, global_step=step)

    def log_config(self, config: Dict, config_file_name: str):
        super().log_config(config, config_file_name)

    def close(self):
        self.writer.close()
