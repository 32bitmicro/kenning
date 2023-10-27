# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A wrapper for the TVM runtime of the YOLOv3 algorithm.

This ModelWrapper handles specific outputs to the YOLOv3
model compiled directly using TVM framework.
Except for the actual model output, there is
additional metadata from the CFG model definition stored in the outputs
from TVM-compiled model.
"""
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.modelwrappers.object_detection.yolo_wrapper import YOLOWrapper


class TVMDarknetCOCOYOLOV3(YOLOWrapper):
    """
    Model wrapper for TVM-compiled YOLOv3 model.
    """

    pretrained_model_uri = "kenning:///models/object_detection/yolov3.cfg"
    default_dataset = COCODataset2017
    arguments_structure = {}

    @classmethod
    def _get_io_specification(cls, keyparams, batch_size):
        return {
            "input": [
                {
                    "name": "data",
                    "shape": (
                        batch_size,
                        3,
                        keyparams["width"],
                        keyparams["height"],
                    ),
                    "dtype": "float32",
                }
            ],  # noqa: E501
            "output": [],
        }

    def get_output_formats(self):
        return ["darknet"]

    def get_framework_and_version(self):
        return ("darknet", "alexeyab")
