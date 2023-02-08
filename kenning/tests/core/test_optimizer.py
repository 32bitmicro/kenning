import pytest
from typing import Type, Final
import os

from kenning.core.optimizer import Optimizer
from kenning.core.optimizer import ConversionError
from kenning.core.optimizer import CompilationError
from kenning.compilers import *  # noqa: 401, 403
from kenning.tests.core.conftest import get_tmp_name
from kenning.tests.core.conftest import get_all_subclasses
from kenning.tests.core.conftest import get_default_dataset_model


OPTIMIZER_SUBCLASSES: Final = get_all_subclasses(Optimizer)

OPTIMIZER_INPUTTYPES: Final = [
    (opt, inp) for opt in OPTIMIZER_SUBCLASSES for inp in opt.inputtypes
]


class TestOptimizer:
    @pytest.mark.parametrize('opt_cls,inputtype', [
        pytest.param(opt_cls, inputtype, marks=[
            pytest.mark.dependency(
                name=f'test_initializer[{opt_cls.__name__}]'
            ),
            pytest.mark.xdist_group(name=f'TestOptimizer_{opt_cls.__name__}')
        ])
        for opt_cls, inputtype in OPTIMIZER_INPUTTYPES
    ])
    def test_initializer(self, opt_cls: Type[Optimizer], inputtype: str):
        compiled_model_path = get_tmp_name()
        dataset, _ = get_default_dataset_model(inputtype)

        _ = opt_cls(dataset, compiled_model_path)

    @pytest.mark.parametrize('opt_cls,inputtype', [
        pytest.param(opt_cls, inputtype, marks=[
            pytest.mark.dependency(
                depends=[f'test_initializer[{opt_cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestOptimizer_{opt_cls.__name__}')
        ])
        for opt_cls, inputtype in OPTIMIZER_INPUTTYPES
    ])
    def test_compilation(self, opt_cls: Type[Optimizer], inputtype: str):
        compiled_model_path = get_tmp_name()
        dataset, model = get_default_dataset_model(inputtype)

        optimizer = opt_cls(dataset, compiled_model_path)
        optimizer.set_input_type(inputtype)
        try:
            optimizer.compile(model.modelpath)
            assert os.path.isfile(compiled_model_path)
        except CompilationError as e:
            pytest.xfail(f'compilation error {e}')
        except ConversionError as e:
            pytest.xfail(f'conversion error {e}')
