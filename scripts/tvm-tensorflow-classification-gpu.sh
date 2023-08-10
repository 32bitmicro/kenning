#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning.scenarios.inference_tester \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --measurements ./build/local-gpu-tvm-tensorflow-classification.json \
    --compiler-cls kenning.optimizers.tvm.TVMCompiler \
    --runtime-cls kenning.runtimes.tvm.TVMRuntime \
    --model-path kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --target "cuda" \
    --compiled-model-path ./build/compiled-model.tar \
    --opt-level 3 \
    --save-model-path ./build/compiled-model.tar \
    --target-device-context cuda \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --verbosity INFO
