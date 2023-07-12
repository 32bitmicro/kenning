#!/bin/bash

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

python -m kenning.scenarios.inference_tester \
    --json-cfg ./scripts/jsonconfigs/yolact-tvm-gpu-detection.json \
    --measurements ./build/yolact-tvm.json \
    --verbosity INFO

python -m kenning.scenarios.inference_tester \
    --json-cfg ./scripts/jsonconfigs/yolact-tflite-detection.json \
    --measurements ./build/yolact-tflite.json \
    --verbosity INFO

python -m kenning.scenarios.render_report \
    --report-path build/yolact-report/report.md \
    --report-name "YOLACT detection report" \
    --root-dir build/yolact-report \
    --img-dir build/yolact-report/imgs \
    --report-types performance detection \
    --measurements build/yolact-tvm.json build/yolact-tflite.json
