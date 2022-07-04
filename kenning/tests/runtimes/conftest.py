import pytest
import uuid
import torch
from kenning.compilers.tvm import TVMCompiler
from pathlib import Path
from torch.autograd import Variable
from typing import Dict, Tuple
from pytest import FixtureRequest


def create_onnx_model(path: Path) -> Tuple[Path, Dict[str, Tuple[int, ...]]]:
    """
    Creates simple convolutional onnx model at given path.

    Parameters
    ----------
    path: Path
        The path to folder where model will be located

    Returns
    -------
    Tuple[Path, Dict[str, Tuple[int, ...]]]:
        The tuple containing path to created Onnx model
        and inputshapes
    """

    modelname = str(uuid.uuid4().hex) + ".onnx"

    model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 3))
    for param in model.parameters():
        param.requires_grad = False
    model[0].weight[0, 0, 0, 0] = 0.0
    model[0].weight[0, 0, 0, 1] = -1.0
    model[0].weight[0, 0, 0, 2] = 0.0
    model[0].weight[0, 0, 1, 0] = -1.0
    model[0].weight[0, 0, 1, 1] = 4.0
    model[0].weight[0, 0, 1, 2] = -1.0
    model[0].weight[0, 0, 2, 0] = 0.0
    model[0].weight[0, 0, 2, 1] = -1.0
    model[0].weight[0, 0, 2, 2] = 0.0
    model[0].bias[:] = 0.0
    data = Variable(torch.FloatTensor(
        [[[[i for i in range(5)] for _ in range(5)]]])
                    )

    modelpath = path / modelname
    torch.onnx.export(model, data, modelpath,
                      input_names=['input.1'], verbose=True)
    return (modelpath, {'input.1': (1, 1, 5, 5)})


@pytest.fixture(scope='function')
def runtimemodel(request: FixtureRequest, tmpfolder: Path) -> Path:
    """
    Fixture that creates simple, runtime specific model.

    In order to use fixture, add: `@pytest.mark.usefixtures('runtimemodel')`
    to testing class

    The optimizer class for model compiling should be passed using:
    `@pytest.mark.parametrize('runtimemodel', [Optimizer], indirect=True)`

    The returned Path object can be accessed using `self.runtimemodel`

    Parameters
    ----------
    request: RequestFixture
        Fixture that allows get Optimizer class through parameter
        and share returned object with testing class
    tmpfolder: Path
        Fixture that provides temporary folder.

    Returns
    -------
    Path:
        The path to created model for runtime
    """

    onnxmodel, inputshapes = create_onnx_model(tmpfolder)
    compiledmodelname = (uuid.uuid4().hex)
    if issubclass(request.param, TVMCompiler):
        compiledmodelname += '.so'
    compiledmodelpath = tmpfolder / compiledmodelname
    optimizer = request.param(None, compiledmodelpath)
    optimizer.compile(onnxmodel, inputshapes, dtype='float32')
    request.cls.runtimemodel = compiledmodelpath
