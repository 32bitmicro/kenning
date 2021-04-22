#!/usr/bin/env python

"""
The sample benchmark for classification problem.

It works with Imagenet-trained models, provides 224x224x3 float tensors as
numpy arrays. It also expects 1000-element float vector as an output.

It provides random data, so it is not applicable for the quality measures.
This test is only for performance tests.
"""

import sys
import argparse
from pathlib import Path
import json
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path

from edge_ai_tester.core.model import ModelWrapper
from edge_ai_tester.utils.class_loader import load_class
from edge_ai_tester.core.measurements import MeasurementsCollector
from edge_ai_tester.core.measurements import systemstatsmeasurements
from edge_ai_tester.utils import logger
from edge_ai_tester.core.report import create_report_from_measurements
from edge_ai_tester.resources import reports
from edge_ai_tester.core.drawing import create_line_plot
from edge_ai_tester.core.drawing import draw_confusion_matrix


@systemstatsmeasurements('full_run_statistics')
def test_inference(modelwrapper: ModelWrapper):
    """
    Benchmarks inference for a given model.

    Parameters
    ----------
    modelwrapper : ModelWrapper
        Model wrapper object with given dataset and configuration

    Returns
    -------
    Measurements : the benchmark results
    """

    frameworktuple = modelwrapper.get_framework_and_version()

    MeasurementsCollector.measurements += {
        'framework': frameworktuple[0],
        'version': frameworktuple[1]
    }

    return modelwrapper.test_inference()


def main(argv):
    parser = argparse.ArgumentParser(argv[0], add_help=False)
    parser.add_argument(
        'modelwrappercls',
        help='ModelWrapper-based class with inference implementation to import',  # noqa: E501
    )
    parser.add_argument(
        'datasetcls',
        help='Dataset-based class with dataset to import',
    )
    parser.add_argument(
        'output',
        help='The path to the output directory',
        type=Path
    )
    parser.add_argument(
        'reportname',
        help='The name of the report, used as RST name and resources prefix',
        type=str
    )
    parser.add_argument(
        '--resources-dir',
        help='The path to the directory with resources',
        type=Path
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )

    args, _ = parser.parse_known_args(argv[1:])

    modelwrappercls = load_class(args.modelwrappercls)
    datasetcls = load_class(args.datasetcls)

    parser = argparse.ArgumentParser(
        argv[0],
        parents=[
            parser,
            modelwrappercls.form_argparse()[0],
            datasetcls.form_argparse()[0]
        ]
    )

    args = parser.parse_args(argv[1:])

    if args.resources_dir is None:
        args.resources_dir = Path(args.output / 'img')

    args.output.mkdir(parents=True, exist_ok=True)
    args.resources_dir.mkdir(parents=True, exist_ok=True)

    logger.set_verbosity(args.verbosity)
    logger.get_logger()

    dataset = datasetcls.from_argparse(args)
    inferenceobj = modelwrappercls.from_argparse(dataset, args)

    test_inference(inferenceobj)

    reportname = args.reportname

    measurementsdata = MeasurementsCollector.measurements.data

    MeasurementsCollector.measurements.data['eval_confusion_matrix'] = MeasurementsCollector.measurements.data['eval_confusion_matrix'].tolist()  # noqa: E501

    with open(args.output, 'w') as measurementsfile:
        json.dump(
            MeasurementsCollector.measurements.data,
            measurementsfile,
            indent=2
        )


if __name__ == '__main__':
    main(sys.argv)
