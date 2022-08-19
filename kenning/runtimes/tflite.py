"""
Runtime implementation for TFLite models.
"""

from pathlib import Path
import numpy as np
from typing import Optional, List, Union

from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol


class TFLiteRuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on TFLite models.
    """

    supported_types = ['float32', 'int8', 'uint8']

    arguments_structure = {
        'modelpath': {
            'argparse_name': '--save-model-path',
            'description': 'Path where the model will be uploaded',
            'type': Path,
            'default': 'model.tar'
        },
        'inputdtype': {
            'argparse_name': '--input-dtype',
            'description': 'Type of input tensor elements',
            'enum': supported_types,
            'default': 'float32',
            'is_list': True
        },
        'outputdtype': {
            'argparse_name': '--output-dtype',
            'description': 'Type of output tensor elements',
            'enum': supported_types,
            'default': 'float32',
            'is_list': True
        },
        'delegates': {
            'argparse_name': '--delegates-list',
            'description': 'List of runtime delegates for the TFLite runtime',
            'default': None,
            'is_list': True,
            'nullable': True
        }
    }

    def __init__(
            self,
            protocol: RuntimeProtocol,
            modelpath: Path,
            inputdtype: Union[str, List[str]] = 'float32',
            outputdtype: Union[str, List[str]] = 'float32',
            delegates: Optional[List] = None,
            collect_performance_data: bool = True):
        """
        Constructs TFLite Runtime pipeline.

        Parameters
        ----------
        protocol : RuntimeProtocol
            Communication protocol
        modelpath : Path
            Path for the model file.
        inputdtype : Union[str, List[str]]
            Type of the input data
        outputdtype : Union[str, List[str]]
            Type of the output data
        delegates : List
            List of TFLite acceleration delegate libraries
        """
        self.modelpath = modelpath
        self.interpreter = None
        self.inputdtype = inputdtype
        self.outputdtype = outputdtype
        self.delegates = delegates
        super().__init__(protocol, collect_performance_data)

    @classmethod
    def from_argparse(cls, protocol, args):
        return cls(
            protocol,
            args.save_model_path,
            args.input_dtype,
            args.output_dtype,
            args.delegates_list,
            args.disable_performance_measurements
        )

    def prepare_model(self, input_data):
        try:
            import tflite_runtime.interpreter as tflite
        except ModuleNotFoundError:
            from tensorflow import lite as tflite
        self.log.info('Loading model')
        if input_data:
            with open(self.modelpath, 'wb') as outmodel:
                outmodel.write(input_data)
        delegates = None
        if self.delegates:
            delegates = [tflite.load_delegate(delegate) for delegate in self.delegates]  # noqa: E501
        self.interpreter = tflite.Interpreter(
            str(self.modelpath),
            experimental_delegates=delegates,
            num_threads=4
        )
        self.interpreter.allocate_tensors()
        self.signature = self.interpreter.get_signature_runner()
        self.sginfo = self.interpreter.get_signature_list()['serving_default']

        if isinstance(self.outputdtype, str):
            self.outputdtype = [
                self.outputdtype for _ in self.interpreter.get_output_details()
            ]
        self.outputdtype = [np.dtype(dt) for dt in self.outputdtype]
        if isinstance(self.inputdtype, str):
            self.inputdtype = [
                self.inputdtype for _ in self.interpreter.get_input_details()
            ]
        self.inputdtype = [np.dtype(dt) for dt in self.inputdtype]

        self.log.info('Model loading ended successfully')
        return True

    def prepare_input(self, input_data):
        self.log.debug(f'Preparing inputs of size {len(input_data)}')
        self.inputs = {}
        input_names = self.sginfo['inputs']
        for datatype, name in zip(self.inputdtype, input_names):
            model_details = self.signature.get_input_details()[name]
            expected_size = np.prod(model_details['shape']) * datatype.itemsize
            input = np.frombuffer(input_data[:expected_size], dtype=datatype)
            try:
                input = input.reshape(model_details['shape'])
                input_size = np.prod(input.shape) * datatype.itemsize
                if expected_size != input_size:
                    self.log.error(f'Invalid input size:  {expected_size} != {input_size}')  # noqa E501
                    raise ValueError
                scale, zero_point = model_details['quantization']
                if scale != 0 and zero_point != 0:
                    input = (input / scale + zero_point).astype(model_details['dtype']) # noqa E501
                self.inputs[name] = input.astype(model_details['dtype'])
                input_data = input_data[expected_size:]
            except ValueError as ex:
                self.log.error(f'Failed to load input: {ex}')
                return False
        return True

    def run(self):
        self.outputs = self.signature(**self.inputs)

    def upload_output(self, input_data):
        self.log.debug('Uploading output')
        result = bytes()
        output_names = self.sginfo['outputs']
        for datatype, name in zip(self.outputdtype, output_names):
            model_details = self.signature.get_output_details()[name]
            output = self.outputs[name]
            if datatype != model_details['dtype']:
                scale, zero_point = model_details['quantization']
                if scale != 0 and zero_point != 0:
                    output = (output.astype(np.float32) - zero_point) * scale
                output = output.astype(datatype)
            result += output.tobytes()
        return result
