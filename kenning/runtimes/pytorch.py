"""
Runtime implementation for PyTorch models
"""
from typing import Optional, List
from pathlib import Path

from kenning.core.runtime import (
    InputNotPreparedError,
    ModelNotPreparedError,
    Runtime,
)
from kenning.core.runtimeprotocol import RuntimeProtocol


class PyTorchRuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on PyTorch models.
    """

    arguments_structure = {
        "modelpath": {
            "argparse_name": "--save-model-path",
            "description": "Path where the model will be uploaded",
            "type": Path,
            "default": "model.pth",
        }
    }

    def __init__(
        self,
        protocol: RuntimeProtocol,
        modelpath: Path,
        collect_performance_data: bool = True,
    ):
        """
        Constructs PyTorch runtime

        Parameters
        ----------
        protocol: RuntimeProtocol
            The implementation of the host-target communication protocol
        modelpath: Path
            Path for the model file
        collect_performance_data: bool
            Disable collection and processing of performance metrics
        """
        import torch

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.modelpath = modelpath
        self.model = None
        self.input: Optional[List] = None
        self.output: Optional[List] = None
        super().__init__(protocol, collect_performance_data)

    def prepare_model(self, input_data: Optional[bytes]) -> bool:
        self.log.info("Loading model")
        import torch

        if input_data:
            with open(self.modelpath, "wb") as fd:
                fd.write(input_data)

        self.model = torch.load(self.modelpath, map_location=self.device)
        if isinstance(self.model, torch.nn.Module):
            self.model = torch.jit.script(self.model.eval())
        elif not isinstance(self.model, torch.jit.ScriptModule):
            self.log.error(
                f"Loaded model is type {type(self.model).__name__}"
                ", only torch.nn.Module and torch.jit.ScriptModule"
                " supported"
            )
            return False
        self.model = torch.jit.freeze(self.model)
        self.log.info("Model loading ended successfully")
        return True

    def prepare_input(self, input_data: bytes):
        self.log.debug(f"Preparing inputs of size {len(input_data)}")
        import torch

        try:
            self.input = self.preprocess_input(input_data)
        except ValueError as ex:
            self.log.error(f"Failed to load input: {ex}")
            return False

        for id, input in enumerate(self.input):
            self.input[id] = torch.from_numpy(input.copy()).to(self.device)
        return True

    def run(self):
        if self.model is None:
            raise ModelNotPreparedError
        if self.input is None:
            raise InputNotPreparedError
        import torch

        with torch.no_grad():
            self.output = [self.model(data) for data in self.input]
        self.input = None

    def upload_output(self, input_data):
        self.log.debug("Uploading output")
        if self.model is None:
            raise ModelNotPreparedError
        import torch

        for id, output in enumerate(self.output):
            if isinstance(output, torch.Tensor):
                self.output[id] = output.cpu().numpy()

        return self.postprocess_output(self.output)

    @classmethod
    def from_argparse(cls, protocol, args):
        return cls(protocol, args.save_model_path)