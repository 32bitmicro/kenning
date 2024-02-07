#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script that generates report files based on Measurements JSON output.

It requires providing the report type and JSON file to extract data from.
"""

import argparse
import copy
import json
import re
import sys
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from argcomplete import DirectoriesCompleter, FilesCompleter

if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path

from kenning.cli.command_template import (
    DEFAULT_GROUP,
    GROUP_SCHEMA,
    REPORT,
    TEST,
    ArgumentsGroups,
    CommandTemplate,
)
from kenning.core.metrics import (
    compute_classification_metrics,
    compute_detection_metrics,
    compute_performance_metrics,
    compute_renode_metrics,
)
from kenning.resources import reports
from kenning.utils.class_loader import get_command
from kenning.utils.logger import KLogger
from kenning.utils.pipeline_runner import UNOPTIMIZED_MEASUREMENTS

# REPORT_TYPES:
PERFORMANCE = "performance"
CLASSIFICATION = "classification"
DETECTION = "detection"
RENODE = "renode_stats"
REPORT_TYPES = [PERFORMANCE, CLASSIFICATION, DETECTION, RENODE]


def get_model_name(filepath: Path) -> str:
    """
    Generates the name of the model. Path to the measurements file is used for
    name generation.

    Parameters
    ----------
    filepath : Path
        Path to the measurements file.

    Returns
    -------
    str
        Name of the model used when generating the report.
    """
    return str(filepath).replace("/", ".")


def get_plot_wildcard_path(plot_path: Path, root_dir: Path) -> str:
    """
    Generate wildcard plot path relative to given directory wshich can be used
    int report.

    Parameters
    ----------
    plot_path : Path
        Path to the saved plot.
    root_dir : Path
        Report root directory.

    Returns
    -------
    str
        Universal plot path relative to report root directory.
    """
    return str(
        plot_path.with_suffix(plot_path.suffix + ".*").relative_to(root_dir)
    )


def performance_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    color_offset: int = 0,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates performance section of the report.

    Parameters
    ----------
    measurementsdata : Dict[str, Any]
        Statistics from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    imgprefix : str
        Prefix to the image file name.
    root_dir : Path
        Path to the root of the documentation project involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    color_offset : int
        How many colors from default color list should be skipped.
    draw_titles : bool
        Should titles be drawn on the plot.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from servis import render_time_series_plot_with_histogram

    from kenning.core.drawing import SERVIS_PLOT_OPTIONS
    from kenning.core.report import create_report_from_measurements

    KLogger.info(
        f'Running performance_report for {measurementsdata["model_name"]}'
    )
    metrics = compute_performance_metrics(measurementsdata)
    measurementsdata |= metrics

    # Shifting colors to match color_offset
    plot_options = copy.deepcopy(SERVIS_PLOT_OPTIONS)
    plot_options["colormap"] = plot_options["colormap"][color_offset:]

    inference_step = None
    if "target_inference_step" in measurementsdata:
        KLogger.info("Using target measurements for inference time")
        inference_step = "target_inference_step"
    elif "protocol_inference_step" in measurementsdata:
        KLogger.info("Using protocol measurements for inference time")
        inference_step = "protocol_inference_step"
    else:
        KLogger.warning("No inference time measurements in the report")

    if inference_step:
        plot_path = imgdir / f"{imgprefix}inference_time"
        render_time_series_plot_with_histogram(
            ydata=measurementsdata[inference_step],
            xdata=measurementsdata[f"{inference_step}_timestamp"],
            title="Inference time" if draw_titles else None,
            xtitle="Time",
            xunit="s",
            ytitle="Inference time",
            yunit="s",
            outpath=str(plot_path),
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )

        measurementsdata["inferencetimepath"] = get_plot_wildcard_path(
            plot_path, root_dir
        )

        measurementsdata["inferencetime"] = measurementsdata[inference_step]

    if "session_utilization_mem_percent" in measurementsdata:
        KLogger.info("Using target measurements memory usage percentage")
        plot_path = imgdir / f"{imgprefix}cpu_memory_usage"
        render_time_series_plot_with_histogram(
            ydata=measurementsdata["session_utilization_mem_percent"],
            xdata=measurementsdata["session_utilization_timestamp"],
            title="Memory usage" if draw_titles else None,
            xtitle="Time",
            xunit="s",
            ytitle="Memory usage",
            yunit="%",
            outpath=str(plot_path),
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )

        measurementsdata["memusagepath"] = get_plot_wildcard_path(
            plot_path, root_dir
        )
    else:
        KLogger.warning("No memory usage measurements in the report")

    if "session_utilization_cpus_percent" in measurementsdata:
        KLogger.info("Using target measurements CPU usage percentage")
        plot_path = imgdir / f"{imgprefix}cpu_usage"
        render_time_series_plot_with_histogram(
            ydata=measurementsdata["session_utilization_cpus_percent_avg"],
            xdata=measurementsdata["session_utilization_timestamp"],
            title="Average CPU usage" if draw_titles else None,
            xtitle="Time",
            xunit="s",
            ytitle="Average CPU usage",
            yunit="%",
            outpath=str(plot_path),
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )

        measurementsdata["cpuusagepath"] = get_plot_wildcard_path(
            plot_path, root_dir
        )
    else:
        KLogger.warning("No memory usage measurements in the report")

    if "session_utilization_gpu_mem_utilization" in measurementsdata:
        KLogger.info("Using target measurements GPU memory usage percentage")
        plot_path = imgdir / f"{imgprefix}gpu_memory_usage"
        gpumemmetric = "session_utilization_gpu_mem_utilization"
        if len(measurementsdata[gpumemmetric]) == 0:
            KLogger.warning(
                "Incorrectly collected data for GPU memory utilization"
            )
        else:
            render_time_series_plot_with_histogram(
                ydata=measurementsdata[gpumemmetric],
                xdata=measurementsdata["session_utilization_gpu_timestamp"],
                title="GPU memory usage" if draw_titles else None,
                xtitle="Time",
                xunit="s",
                ytitle="GPU memory usage",
                yunit="%",
                outpath=str(plot_path),
                skipfirst=True,
                outputext=image_formats,
                **plot_options,
            )

            measurementsdata["gpumemusagepath"] = get_plot_wildcard_path(
                plot_path, root_dir
            )
    else:
        KLogger.warning("No GPU memory usage measurements in the report")

    if "session_utilization_gpu_utilization" in measurementsdata:
        KLogger.info("Using target measurements GPU utilization")
        plot_path = imgdir / f"{imgprefix}gpu_usage"
        if len(measurementsdata["session_utilization_gpu_utilization"]) == 0:
            KLogger.warning("Incorrectly collected data for GPU utilization")
        else:
            render_time_series_plot_with_histogram(
                ydata=measurementsdata["session_utilization_gpu_utilization"],
                xdata=measurementsdata["session_utilization_gpu_timestamp"],
                title="GPU Utilization" if draw_titles else None,
                xtitle="Time",
                xunit="s",
                ytitle="Utilization",
                yunit="%",
                outpath=str(plot_path),
                skipfirst=True,
                outputext=image_formats,
                **plot_options,
            )

            measurementsdata["gpuusagepath"] = get_plot_wildcard_path(
                plot_path, root_dir
            )
    else:
        KLogger.warning("No GPU utilization measurements in the report")

    with path(reports, "performance.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, measurementsdata
        )


def comparison_performance_report(
    measurementsdata: List[Dict],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List] = None,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates performance comparison section of report.

    Parameters
    ----------
    measurementsdata : List[Dict]
        Statistics of every model from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    root_dir : Path
        Path to the root of the documentation project involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    colors : Optional[List]
        Colors to be used in the plots.
    draw_titles : bool
        Should titles be drawn on the plot.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from servis import render_multiple_time_series_plot

    from kenning.core.drawing import (
        SERVIS_PLOT_OPTIONS,
        RadarChart,
        ViolinComparisonPlot,
    )
    from kenning.core.report import create_report_from_measurements

    KLogger.info("Running comparison_performance_report")

    metric_names = {
        "inference_step": ("Inference time", "s"),
        "session_utilization_mem_percent": ("Memory usage", "%"),
        "session_utilization_cpus_percent_avg": ("CPU usage", "%"),
        "session_utilization_gpu_mem_utilization": ("GPU memory usage", "%"),
        "session_utilization_gpu_utilization": ("GPU usage", "%"),
    }
    common_metrics = set(metric_names.keys())
    hardware_usage_metrics = common_metrics - {"inference_step"}
    report_variables = {
        "report_name": measurementsdata[0]["report_name"],
        "report_name_simple": measurementsdata[0]["report_name_simple"],
    }

    for data in measurementsdata:
        if "target_inference_step" in data:
            data["inference_step"] = data["target_inference_step"]
            data["inference_step_timestamp"] = data[
                "target_inference_step_timestamp"
            ]
        elif "protocol_inference_step" in data:
            data["inference_step"] = data["protocol_inference_step"]
            data["inference_step_timestamp"] = data[
                "protocol_inference_step_timestamp"
            ]

        if "session_utilization_cpus_percent" in data:
            metrics = compute_performance_metrics(data)
            data["session_utilization_cpus_percent_avg"] = metrics[
                "session_utilization_cpus_percent_avg"
            ]

        gpumetrics = [
            "session_utilization_gpu_mem_utilization",
            "session_utilization_gpu_utilization",
        ]
        for gpumetric in gpumetrics:
            if gpumetric in data and len(data[gpumetric]) == 0:
                del data[gpumetric]

        modelmetrics = set(data.keys())
        common_metrics &= modelmetrics

    for metric, (metric_name, unit) in metric_names.items():
        metric_data = {}
        if metric_name == "Inference time":
            timestamp_key = "inference_step_timestamp"
        elif metric_name in ("GPU usage", "GPU memory usage"):
            timestamp_key = "session_utilization_gpu_timestamp"
        else:
            timestamp_key = "session_utilization_timestamp"
        if timestamp_key not in data:
            KLogger.warning(
                f'Missing measurement "{timestamp_key}" in the measurements '
                f"file. Can't provide benchmarks for {metric_name}"
            )
            continue
        timestamps = {
            data["model_name"]: data[timestamp_key]
            for data in measurementsdata
        }

        for data in measurementsdata:
            if metric in data:
                metric_data[data["model_name"]] = data[metric]
        if len(metric_data) > 1:
            plot_path = imgdir / f"{metric}_comparison"
            render_multiple_time_series_plot(
                ydatas=[list(metric_data.values())],
                xdatas=[list(timestamps.values())],
                title=f"{metric_name} comparison" if draw_titles else None,
                subtitles=None,
                xtitles=["Time"],
                xunits=["s"],
                ytitles=[metric_name],
                yunits=[unit],
                legend_labels=list(metric_data.keys()),
                outpath=plot_path,
                outputext=image_formats,
                **SERVIS_PLOT_OPTIONS,
            )
            report_variables[f"{metric}_path"] = get_plot_wildcard_path(
                plot_path, root_dir
            )

    common_metrics = sorted(list(common_metrics))
    visualizationdata = {}
    for data in measurementsdata:
        visualizationdata[data["model_name"]] = [
            data[metric] for metric in common_metrics
        ]

    plot_path = imgdir / "mean_performance_comparison"
    ViolinComparisonPlot(
        title="Performance comparison plot" if draw_titles else None,
        metric_data=visualizationdata,
        metric_labels=[
            f"{metric_names[metric][0]} [{metric_names[metric][1]}]"
            for metric in common_metrics
        ],
        colors=colors,
    ).plot(plot_path, image_formats)
    report_variables["meanperformancepath"] = get_plot_wildcard_path(
        plot_path, root_dir
    )

    hardware_usage_metrics = sorted(list(hardware_usage_metrics))
    measurements_metrics = set()
    for data in measurementsdata:
        measurements_metrics = measurements_metrics.union(data.keys())

    if set(hardware_usage_metrics).intersection(measurements_metrics):
        usage_visualization = {}
        for data in measurementsdata:
            usage_visualization[data["model_name"]] = [
                np.mean(data.get(metric, 0)) / 100
                for metric in hardware_usage_metrics
            ]

        plot_path = imgdir / "hardware_usage_comparison"
        RadarChart(
            title="Resource usage comparison" if draw_titles else None,
            metric_data=usage_visualization,
            metric_labels=[
                metric_names[metric][0] for metric in hardware_usage_metrics
            ],
            colors=colors,
        ).plot(plot_path, image_formats)
        report_variables["hardwareusagepath"] = get_plot_wildcard_path(
            plot_path, root_dir
        )

    with path(reports, "performance_comparison.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        )


def classification_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    cmap: Optional[Any] = None,
    colors: Optional[List] = None,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates classification quality section of the report.

    Parameters
    ----------
    measurementsdata : Dict[str, Any]
        Statistics from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    imgprefix : str
        Prefix to the image file name.
    root_dir : Path
        Path to the root of the documentation project involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    cmap : Optional[Any]
        Color map to be used in the plots (matplotlib.colors.ListedColormap)
    colors : Optional[List]
        Colors to be used in the plots.
    draw_titles : bool
        Should titles be drawn on the plot.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from kenning.core.drawing import Barplot, ConfusionMatrixPlot
    from kenning.core.report import create_report_from_measurements

    KLogger.info(
        f'Running classification report for {measurementsdata["model_name"]}'
    )
    metrics = compute_classification_metrics(measurementsdata)
    measurementsdata |= metrics

    if "eval_confusion_matrix" in measurementsdata:
        KLogger.info("Using confusion matrix")
        confusion_path = imgdir / f"{imgprefix}confusion_matrix"
        ConfusionMatrixPlot(
            title="Confusion matrix" if draw_titles else None,
            confusion_matrix=measurementsdata["eval_confusion_matrix"],
            class_names=measurementsdata["class_names"],
            cmap=cmap,
        ).plot(confusion_path, image_formats)
        measurementsdata["confusionpath"] = get_plot_wildcard_path(
            confusion_path, root_dir
        )
    elif "predictions" in measurementsdata:
        KLogger.info("Using predictions")

        predictions = list(
            zip(
                measurementsdata["predictions"],
                measurementsdata["class_names"],
            )
        )
        predictions.sort(key=lambda x: x[0], reverse=True)

        predictions = list(zip(*predictions))

        predictions_path = imgdir / f"{imgprefix}predictions"
        Barplot(
            title="Predictions" if draw_titles else None,
            x_label="Class",
            y_label="Percentage",
            y_unit="%",
            x_data=list(predictions[1]),
            y_data={"predictions": list(predictions[0])},
            colors=colors,
        ).plot(predictions_path, image_formats)
        measurementsdata["predictionspath"] = get_plot_wildcard_path(
            predictions_path, root_dir
        )
    else:
        KLogger.error(
            "Confusion matrix and predictions not present for classification "
            "report"
        )
        return ""

    with path(reports, "classification.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, measurementsdata
        )


def comparison_classification_report(
    measurementsdata: List[Dict],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List] = None,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates classification comparison section of report.

    Parameters
    ----------
    measurementsdata : List[Dict]
        Statistics of every model from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    root_dir : Path
        Path to the root of the documentation project involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    colors : Optional[List]
        Colors to be used in the plots.
    draw_titles : bool
        Should titles be drawn on the plot.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from kenning.core.drawing import Barplot, BubblePlot, RadarChart
    from kenning.core.report import create_report_from_measurements

    KLogger.info("Running comparison_classification_report")
    # HTML plots format unsupported, removing html

    # check that each measurements have the same classes
    for data in measurementsdata:
        assert (
            measurementsdata[0]["class_names"] == data["class_names"]
        ), "Invalid class names in measurements"

    report_variables = {
        "report_name": measurementsdata[0]["report_name"],
        "report_name_simple": measurementsdata[0]["report_name_simple"],
    }
    names = [data["model_name"] for data in measurementsdata]
    metric_visualization = {}
    accuracy, mean_inference_time, model_sizes = [], [], []
    skip_inference_metrics = False
    for data in measurementsdata:
        performance_metrics = compute_performance_metrics(data)
        if "inferencetime_mean" not in performance_metrics:
            skip_inference_metrics = True
            break

        classification_metrics = compute_classification_metrics(data)
        model_accuracy = classification_metrics["accuracy"]
        model_precision = classification_metrics["mean_precision"]
        model_sensitivity = classification_metrics["mean_sensitivity"]
        accuracy.append(model_accuracy)

        model_inferencetime_mean = performance_metrics["inferencetime_mean"]
        mean_inference_time.append(model_inferencetime_mean)

        if "compiled_model_size" in data:
            model_sizes.append(data["compiled_model_size"])
        else:
            KLogger.warning(
                "Missing information about model size in measurements"
                " - computing size based on average RAM usage"
            )
            model_sizes.append(
                performance_metrics["session_utilization_mem_percent_mean"]
            )

        # Accuracy, precision, sensitivity
        metric_visualization[data["model_name"]] = [
            model_accuracy,
            model_precision,
            model_sensitivity,
        ]

    if not skip_inference_metrics:
        plot_path = imgdir / "accuracy_vs_inference_time"
        BubblePlot(
            title="Accuracy vs Mean inference time" if draw_titles else None,
            x_data=mean_inference_time,
            x_label="Mean inference time [s]",
            y_data=accuracy,
            y_label="Accuracy",
            size_data=model_sizes,
            size_label="Model size",
            bubble_labels=names,
            colors=colors,
        ).plot(plot_path, image_formats)
        report_variables["bubbleplotpath"] = get_plot_wildcard_path(
            plot_path, root_dir
        )

        plot_path = imgdir / "classification_metric_comparison"
        RadarChart(
            title="Metric comparison" if draw_titles else None,
            metric_data=metric_visualization,
            metric_labels=["Accuracy", "Mean precision", "Mean recall"],
            colors=colors,
        ).plot(plot_path, image_formats)
        report_variables["radarchartpath"] = get_plot_wildcard_path(
            plot_path, root_dir
        )
        report_variables["model_names"] = names
        report_variables = {
            **report_variables,
            **metric_visualization,
        }

    if "predictions" in measurementsdata[0]:
        predictions = [measurementsdata[0]["class_names"]] + [
            data["predictions"] for data in measurementsdata
        ]
        predictions = list(zip(*predictions))
        predictions.sort(key=lambda x: (sum(x[1:]), x[0]), reverse=True)
        predictions = list(zip(*predictions))
        predictions_data = {
            name: data for name, data in zip(names, predictions[1:])
        }
        predictions_batplot_path = imgdir / "predictions"
        Barplot(
            title="Predictions barplot" if draw_titles else None,
            x_label="Class",
            y_label="Percentage",
            y_unit="%",
            x_data=predictions[0],
            y_data=predictions_data,
            colors=colors,
        ).plot(predictions_batplot_path, image_formats)
        report_variables["predictionsbarpath"] = get_plot_wildcard_path(
            predictions_batplot_path, root_dir
        )
    elif skip_inference_metrics:
        KLogger.warning(
            "No inference measurements available, skipping report generation"
        )
        return ""

    with path(reports, "classification_comparison.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        )


def detection_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    color_offset: int = 0,
    cmap: Optional[Any] = None,
    colors: Optional[List] = None,
    draw_titles: bool = True,
) -> str:
    """
    Creates detection quality section of the report.

    Parameters
    ----------
    measurementsdata : Dict[str, Any]
        Statistics from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    imgprefix : str
        Prefix to the image file name.
    root_dir : Path
        Path to the root of the documentation project involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    color_offset : int
        How many colors from default color list should be skipped.
    cmap : Optional[Any]
        Color map to be used in the plots.
    colors : Optional[List]
        Colors to be used in the plots.
    draw_titles : bool
        Should titles be drawn on the plot.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from kenning.core.drawing import (
        LinePlot,
        RecallPrecisionCurvesPlot,
        RecallPrecisionGradients,
        TruePositiveIoUHistogram,
        TruePositivesPerIoURangeHistogram,
    )
    from kenning.core.report import create_report_from_measurements
    from kenning.datasets.helpers.detection_and_segmentation import (
        compute_ap,
        compute_map_per_threshold,
        get_recall_precision,
    )

    KLogger.info(
        f'Running detection report for {measurementsdata["model_name"]}'
    )
    metrics = compute_detection_metrics(measurementsdata)
    measurementsdata |= metrics

    lines = get_recall_precision(measurementsdata, 0.5)

    aps = []
    for line in lines:
        aps.append(compute_ap(line[0], line[1]))

    curve_path = imgdir / f"{imgprefix}recall_precision_curves"
    RecallPrecisionCurvesPlot(
        title="Recall-Precision curves" if draw_titles else None,
        lines=lines,
        class_names=measurementsdata["class_names"],
    ).plot(curve_path, image_formats)
    measurementsdata["curvepath"] = get_plot_wildcard_path(
        curve_path, root_dir
    )

    gradient_path = imgdir / f"{imgprefix}recall_precision_gradients"
    RecallPrecisionGradients(
        title="Average precision plots" if draw_titles else None,
        lines=lines,
        class_names=measurementsdata["class_names"],
        avg_precisions=aps,
        mean_avg_precision=measurementsdata["mAP"],
        cmap=cmap,
    ).plot(gradient_path, image_formats)
    measurementsdata["gradientpath"] = get_plot_wildcard_path(
        gradient_path, root_dir
    )

    tp_iou = []
    all_tp_ious = []

    for i in measurementsdata["class_names"]:
        dets = (
            measurementsdata[f"eval_det/{i}"]
            if f"eval_det/{i}" in measurementsdata
            else []
        )
        det_tp_iou = [i[2] for i in dets if i[1]]
        if len(det_tp_iou) > 0:
            tp_iou.append(sum(det_tp_iou) / len(det_tp_iou))
            all_tp_ious.extend(det_tp_iou)
        else:
            tp_iou.append(0)

    tpiou_path = imgdir / f"{imgprefix}true_positive_iou_histogram"
    TruePositiveIoUHistogram(
        title="Average True Positive IoU values" if draw_titles else None,
        iou_data=tp_iou,
        class_names=measurementsdata["class_names"],
        colors=colors,
        color_offset=color_offset,
    ).plot(tpiou_path, image_formats)
    measurementsdata["tpioupath"] = get_plot_wildcard_path(
        tpiou_path, root_dir
    )

    if len(all_tp_ious) > 0:
        iouhist_path = imgdir / f"{imgprefix}histogram_tp_iou_values"
        TruePositivesPerIoURangeHistogram(
            title="Histogram of True Positive IoU values"
            if draw_titles
            else None,
            iou_data=all_tp_ious,
            colors=colors,
            color_offset=color_offset,
        ).plot(iouhist_path, image_formats)
        measurementsdata["iouhistpath"] = get_plot_wildcard_path(
            iouhist_path, root_dir
        )

    thresholds = np.arange(0.2, 1.05, 0.05)
    mapvalues = compute_map_per_threshold(measurementsdata, thresholds)

    map_path = imgdir / f"{imgprefix}map"
    LinePlot(
        title=(
            "mAP value change over objectness threshold values"
            if draw_titles
            else None
        ),
        x_label="threshold",
        y_label="mAP",
        lines=[(thresholds, mapvalues)],
        colors=colors,
        color_offset=color_offset,
    ).plot(map_path, image_formats)
    measurementsdata["mappath"] = get_plot_wildcard_path(map_path, root_dir)
    measurementsdata["max_mAP"] = max(mapvalues)
    measurementsdata["max_mAP_index"] = thresholds[np.argmax(mapvalues)].round(
        2
    )

    if "eval/recordings" in measurementsdata:
        # Find index of sequence with the best, worst, and average mAP
        total_mAPs = np.array(
            [x["mAP"]["total"] for x in measurementsdata["eval/recordings"]]
        )
        num_plots_to_draw = min(5, len(total_mAPs))
        sorted_mAP_idxs = np.argsort(total_mAPs)
        min_mAP_idxs = sorted_mAP_idxs[:num_plots_to_draw]
        max_mAP_idxs = sorted_mAP_idxs[-num_plots_to_draw:]

        # Line plot for top `n` worst recordings
        map_path = imgdir / f"{imgprefix}map_worst_recordings"
        lines = []
        for index in min_mAP_idxs:
            mAP_data = measurementsdata["eval/recordings"][index]["mAP"]
            mAP_values = [
                float(x) for x in mAP_data["values"][1:-1].strip().split()
            ]
            mAP_thresholds = [
                float(x) for x in mAP_data["thresholds"][1:-1].strip().split()
            ]
            lines.append((mAP_thresholds, mAP_values))
        LinePlot(
            title=f"mAP for {num_plots_to_draw} worst recordings"
            if draw_titles
            else None,
            x_label="threshold",
            y_label="mAP",
            lines=lines,
            lines_labels=[
                measurementsdata["eval/recordings"][idx]["name"]
                for idx in min_mAP_idxs
            ],
            colors=colors,
        ).plot(map_path, image_formats)
        measurementsdata["map_worst_recordings"] = get_plot_wildcard_path(
            map_path, root_dir
        )

        # Line plot for top `n` best recordings
        map_path = imgdir / f"{imgprefix}map_best_recordings"
        lines = []
        for index in max_mAP_idxs:
            mAP_data = measurementsdata["eval/recordings"][index]["mAP"]
            mAP_values = [
                float(x) for x in mAP_data["values"][1:-1].strip().split()
            ]
            mAP_thresholds = [
                float(x) for x in mAP_data["thresholds"][1:-1].strip().split()
            ]
            lines.append((mAP_thresholds, mAP_values))
        LinePlot(
            title=f"mAP for {num_plots_to_draw} best recordings"
            if draw_titles
            else None,
            x_label="threshold",
            y_label="mAP",
            lines=lines,
            lines_labels=[
                measurementsdata["eval/recordings"][idx]["name"]
                for idx in max_mAP_idxs
            ],
            colors=colors,
        ).plot(map_path, image_formats)
        measurementsdata["map_best_recordings"] = get_plot_wildcard_path(
            map_path, root_dir
        )

    with path(reports, "detection.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, measurementsdata
        )


def comparison_detection_report(
    measurementsdata: List[Dict],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List] = None,
    draw_titles: bool = True,
    **kwargs: Any,
) -> str:
    """
    Creates detection comparison section of report.

    Parameters
    ----------
    measurementsdata : List[Dict]
        Statistics of every model from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    root_dir : Path
        Path to the root of the documentation project involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    colors : Optional[List]
        Colors to be used in the plots.
    draw_titles : bool
        Should titles be drawn on the plot.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    KLogger.info("Running comparison_detection_report")

    from kenning.core.drawing import LinePlot
    from kenning.core.report import create_report_from_measurements
    from kenning.datasets.helpers.detection_and_segmentation import (
        compute_map_per_threshold,
    )

    report_variables = {
        "report_name": measurementsdata[0]["report_name"],
        "report_name_simple": measurementsdata[0]["report_name_simple"],
        "model_names": [],
    }

    visualization_data = []
    for data in measurementsdata:
        thresholds = np.arange(0.2, 1.05, 0.05)
        mapvalues = compute_map_per_threshold(data, thresholds)
        visualization_data.append((thresholds, mapvalues))
        report_variables["model_names"].append(data["model_name"])

    plot_path = imgdir / "detection_map_thresholds"
    LinePlot(
        title=(
            "mAP values comparison over different threshold values"
            if draw_titles
            else None
        ),
        x_label="threshold",
        y_label="mAP",
        lines=visualization_data,
        lines_labels=report_variables["model_names"],
        colors=colors,
    ).plot(plot_path, image_formats)
    report_variables["mapcomparisonpath"] = get_plot_wildcard_path(
        plot_path, root_dir
    )

    with path(reports, "detection_comparison.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        )


def renode_stats_report(
    measurementsdata: Dict,
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    draw_titles: bool = True,
    colors: Optional[List] = None,
    color_offset: int = 0,
    **kwargs: Any,
) -> str:
    """
    Creates Renode stats section of the report.

    Parameters
    ----------
    measurementsdata : Dict
        Statistics from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    imgprefix : str
        Prefix to the image file name.
    root_dir : Path
        Path to the root of the documentation project involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    draw_titles : bool
        Should titles be drawn on the plot.
    colors : Optional[List]
        Colors used for plots.
    color_offset : int
        How many colors from default color list should be skipped.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from servis import render_time_series_plot_with_histogram

    from kenning.core.drawing import SERVIS_PLOT_OPTIONS, Barplot, LinePlot
    from kenning.core.report import create_report_from_measurements

    KLogger.info(
        f'Running renode_stats_report for {measurementsdata["model_name"]}'
    )
    # shift colors to match color_offset
    plot_options = copy.deepcopy(SERVIS_PLOT_OPTIONS)
    plot_options["colormap"] = plot_options["colormap"][color_offset:]

    measurementsdata |= compute_renode_metrics([measurementsdata])

    # opcode counter barplot
    if "sorted_opcode_counters" in measurementsdata:
        opcode_counters = measurementsdata["sorted_opcode_counters"]
        instr_barplot_path = imgdir / f"{imgprefix}instr_barplot"

        Barplot(
            title="Instructions barplot" if draw_titles else None,
            x_label="Opcode",
            y_label="Counter",
            x_data=opcode_counters["opcodes"],
            y_data=opcode_counters["counters"],
            colors=colors,
            color_offset=color_offset,
            max_bars_matplotlib=32,
        ).plot(instr_barplot_path, image_formats)
        measurementsdata["instrbarpath"] = get_plot_wildcard_path(
            instr_barplot_path, root_dir
        )

    # vector opcode counter barplot
    if "sorted_vector_opcode_counters" in measurementsdata:
        vector_opcode_counters = measurementsdata[
            "sorted_vector_opcode_counters"
        ]
        vector_instr_barplot_path = imgdir / f"{imgprefix}vector_instr_barplot"

        Barplot(
            title="Vector instructions barplot" if draw_titles else None,
            x_label="Opcode",
            y_label="Counter",
            x_data=vector_opcode_counters["opcodes"],
            y_data=vector_opcode_counters["counters"],
            colors=colors,
            color_offset=color_offset,
            max_bars_matplotlib=32,
        ).plot(vector_instr_barplot_path, image_formats)
        measurementsdata["vectorinstrbarpath"] = get_plot_wildcard_path(
            vector_instr_barplot_path, root_dir
        )

    # executed instructions plot
    for cpu, data in measurementsdata["executed_instructions"].items():
        paths = {}

        executed_instructions_plot_path = (
            imgdir / f"{imgprefix}executed_instructions_{cpu}_plot"
        )

        render_time_series_plot_with_histogram(
            ydata=data,
            xdata=measurementsdata["profiler_timestamps"],
            title=f"Executed instructions for {cpu}" if draw_titles else None,
            xtitle="Interval timestamp",
            xunit="s",
            ytitle="Executed instructions",
            yunit="1/s",
            outpath=executed_instructions_plot_path,
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )
        paths["persec"] = get_plot_wildcard_path(
            executed_instructions_plot_path, root_dir
        )

        cum_executed_instructions_plot_path = (
            imgdir / f"{imgprefix}cumulative_executed_instructions_{cpu}_plot"
        )

        LinePlot(
            lines=[(measurementsdata["profiler_timestamps"], np.cumsum(data))],
            title=f"Executed instructions for {cpu}" if draw_titles else None,
            x_label="Interval timestamp",
            x_unit="s",
            y_label="Total executed instructions",
            colors=plot_options["colormap"],
        ).plot(cum_executed_instructions_plot_path, image_formats)
        paths["cumulative"] = get_plot_wildcard_path(
            cum_executed_instructions_plot_path, root_dir
        )

        if "executedinstrplotpath" not in measurementsdata:
            measurementsdata["executedinstrplotpath"] = {}

        measurementsdata["executedinstrplotpath"][cpu] = paths

    # memory accesses plot
    for access_type in ("read", "write"):
        if not len(measurementsdata["memory_accesses"][access_type]):
            continue

        paths = {}

        memory_access_plot_path = (
            imgdir / f"{imgprefix}memory_{access_type}s_plot"
        )

        render_time_series_plot_with_histogram(
            ydata=measurementsdata["memory_accesses"][access_type],
            xdata=measurementsdata["profiler_timestamps"],
            title=f"Memory {access_type}s" if draw_titles else None,
            xtitle="Interval timestamp",
            xunit="s",
            ytitle=f"Memory {access_type}s",
            yunit="1/s",
            outpath=memory_access_plot_path,
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )
        paths["persec"] = get_plot_wildcard_path(
            memory_access_plot_path, root_dir
        )

        cum_memory_access_plot_path = (
            imgdir / f"{imgprefix}cumulative_memory_{access_type}s_plot"
        )

        LinePlot(
            title=f"Memory {access_type}s" if draw_titles else None,
            x_label="Interval timestamp",
            x_unit="s",
            y_label=f"Total memory {access_type}s",
            lines=[
                (
                    measurementsdata["profiler_timestamps"],
                    np.cumsum(
                        measurementsdata["memory_accesses"][access_type]
                    ),
                )
            ],
            colors=plot_options["colormap"],
        ).plot(cum_memory_access_plot_path, image_formats)
        paths["cumulative"] = get_plot_wildcard_path(
            cum_memory_access_plot_path, root_dir
        )

        if "memoryaccessesplotpath" not in measurementsdata:
            measurementsdata["memoryaccessesplotpath"] = {}

        measurementsdata["memoryaccessesplotpath"][access_type] = paths

    # peripheral accesses plot
    for peripheral, measurements in measurementsdata[
        "peripheral_accesses"
    ].items():
        paths = defaultdict(dict)

        for access_type in ("read", "write"):
            if not sum(measurements[access_type]):
                continue

            peripheral_access_plot_path = (
                imgdir / f"{imgprefix}_{peripheral}_{access_type}s_plot"
            )

            render_time_series_plot_with_histogram(
                ydata=measurements[access_type],
                xdata=measurementsdata["profiler_timestamps"],
                title=f"{peripheral} {access_type}s" if draw_titles else None,
                xtitle="Interval timestamp",
                xunit="s",
                ytitle=f"{peripheral} {access_type}s",
                yunit="1/s",
                outpath=peripheral_access_plot_path,
                skipfirst=True,
                outputext=image_formats,
                **plot_options,
            )

            paths[access_type]["persec"] = get_plot_wildcard_path(
                peripheral_access_plot_path, root_dir
            )

            cum_peripheral_access_plot_path = (
                imgdir
                / f"{imgprefix}cumulative_{peripheral}_{access_type}s_plot"
            )

            LinePlot(
                title=f"{peripheral} {access_type}s" if draw_titles else None,
                x_label="Interval timestamp",
                x_unit="s",
                y_label=f"Total {peripheral} {access_type}s",
                lines=[
                    (
                        measurementsdata["profiler_timestamps"],
                        np.cumsum(measurements[access_type]),
                    )
                ],
                colors=plot_options["colormap"],
            ).plot(cum_peripheral_access_plot_path, image_formats)
            paths[access_type]["cumulative"] = get_plot_wildcard_path(
                cum_peripheral_access_plot_path, root_dir
            )

        if len(paths):
            if "peripheralaccessesplotpath" not in measurementsdata:
                measurementsdata["peripheralaccessesplotpath"] = {}
            measurementsdata["peripheralaccessesplotpath"][peripheral] = paths

    # exceptions plot
    if sum(measurementsdata["exceptions"]):
        paths = {}

        exceptions_plot_path = imgdir / f"{imgprefix}exceptions_plot"

        render_time_series_plot_with_histogram(
            ydata=measurementsdata["exceptions"],
            xdata=measurementsdata["profiler_timestamps"],
            title="Exceptions" if draw_titles else None,
            xtitle="Interval timestamp",
            xunit="s",
            ytitle="Exceptions count",
            yunit="1/s",
            outpath=exceptions_plot_path,
            skipfirst=True,
            outputext=image_formats,
            **plot_options,
        )
        paths["persec"] = get_plot_wildcard_path(
            exceptions_plot_path, root_dir
        )
        cum_exceptions_plot_path = (
            imgdir / f"{imgprefix}cumulative_exceptions_plot"
        )

        LinePlot(
            title="Total xceptions" if draw_titles else None,
            x_label="Interval timestamp",
            x_unit="s",
            y_label="Total exceptions",
            y_unit=None,
            lines=[
                (
                    measurementsdata["profiler_timestamps"],
                    np.cumsum(measurementsdata["exceptions"]),
                )
            ],
            colors=plot_options["colormap"],
        ).plot(cum_exceptions_plot_path, image_formats)

        paths["cumulative"] = get_plot_wildcard_path(
            cum_exceptions_plot_path, root_dir
        )

        measurementsdata["exceptionsplotpath"] = paths

    with path(reports, "renode_stats.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, measurementsdata
        )


def comparison_renode_stats_report(
    measurementsdata: List[Dict],
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    color_offset: int = 0,
    draw_titles: bool = True,
    colors: Optional[List] = None,
    **kwargs: Any,
) -> str:
    """
    Creates Renode stats section of the report.

    Parameters
    ----------
    measurementsdata : List[Dict]
        Statistics from the Measurements class.
    imgdir : Path
        Path to the directory for images.
    root_dir : Path
        Path to the root of the documentation project involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to generate plots.
    color_offset : int
        How many colors from default color list should be skipped.
    draw_titles : bool
        Should titles be drawn on the plot.
    colors : Optional[List]
        Colors used for plots.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from servis import render_multiple_time_series_plot

    from kenning.core.drawing import SERVIS_PLOT_OPTIONS, Barplot, LinePlot
    from kenning.core.report import create_report_from_measurements

    def retrieve_non_zero_profiler_data(
        measurementsdata: List[Dict], keys: List[str] = []
    ) -> Tuple[List, List, List]:
        ydata = []
        xdata = []
        labels = []
        for m in measurementsdata:
            data = m
            for k in keys:
                if k in data.keys():
                    data = data[k]
                else:
                    data = None
                    break
            if data is None:
                continue

            if sum(data) == 0:
                continue

            ydata.append(data)
            xdata.append(m["profiler_timestamps"])
            labels.append(m["model_name"])

        return xdata, ydata, labels

    KLogger.info("Running comparison_renode_stats_report")

    report_variables = {
        "report_name": measurementsdata[0]["report_name"],
        "report_name_simple": measurementsdata[0]["report_name_simple"],
    }

    metrics = compute_renode_metrics(measurementsdata)

    # opcode counter barplot
    if "sorted_opcode_counters" in metrics:
        opcode_counters = metrics["sorted_opcode_counters"]
        instr_barplot_path = imgdir / "instr_barplot_comparison"

        Barplot(
            title="Instructions barplot" if draw_titles else None,
            x_label="Opcode",
            y_label="Counter",
            x_data=opcode_counters["opcodes"],
            y_data=opcode_counters["counters"],
            colors=colors,
            max_bars_matplotlib=32,
        ).plot(instr_barplot_path, image_formats)
        report_variables["instrbarpath"] = get_plot_wildcard_path(
            instr_barplot_path, root_dir
        )

    # vector opcode counter barplot
    if "sorted_vector_opcode_counters" in metrics:
        vector_opcode_counters = metrics["sorted_vector_opcode_counters"]
        vector_instr_barplot_path = imgdir / "vector_instr_barplot_comparison"

        Barplot(
            title="Vector instructions barplot" if draw_titles else None,
            x_label="Opcode",
            y_label="Counter",
            x_data=vector_opcode_counters["opcodes"],
            y_data=vector_opcode_counters["counters"],
            colors=colors,
            max_bars_matplotlib=32,
        ).plot(vector_instr_barplot_path, image_formats)

        report_variables["vectorinstrbarpath"] = get_plot_wildcard_path(
            vector_instr_barplot_path, root_dir
        )

    # executed instructions plot
    report_variables["executedinstrplotpath"] = {}

    all_cpus = set()

    for data in measurementsdata:
        all_cpus = all_cpus.union(data["executed_instructions"].keys())

    for cpu in all_cpus:
        xdata, ydata, labels = retrieve_non_zero_profiler_data(
            measurementsdata, ["executed_instructions", cpu]
        )

        paths = {}

        executed_instructions_plot_path = (
            imgdir / f"executed_instructions_{cpu}_plot_comparison"
        )

        render_multiple_time_series_plot(
            ydatas=[ydata],
            xdatas=[xdata],
            title=f"Executed instructions for {cpu} comparison"
            if draw_titles
            else None,
            subtitles=None,
            xtitles=["Interval timestamp"],
            xunits=["s"],
            ytitles=["Executed instructions"],
            yunits=["1/s"],
            legend_labels=labels,
            outpath=executed_instructions_plot_path,
            outputext=image_formats,
            **SERVIS_PLOT_OPTIONS,
        )
        paths["persec"] = get_plot_wildcard_path(
            executed_instructions_plot_path, root_dir
        )

        cum_executed_instructions_plot_path = (
            imgdir / f"cumulative_executed_instructions_{cpu}_plot_comparison"
        )

        LinePlot(
            title=f"Executed instructions for {cpu}" if draw_titles else None,
            x_label="Interval timestamp",
            x_unit="s",
            y_label="Total executed instructions",
            lines=[(x, np.cumsum(y)) for x, y in zip(xdata, ydata)],
            lines_labels=labels,
            colors=SERVIS_PLOT_OPTIONS["colormap"],
        ).plot(cum_executed_instructions_plot_path, image_formats)
        paths["cumulative"] = get_plot_wildcard_path(
            cum_executed_instructions_plot_path, root_dir
        )

        if "executedinstrplotpath" not in report_variables:
            report_variables["executedinstrplotpath"] = {}

        report_variables["executedinstrplotpath"][cpu] = paths

    # memory accesses plot
    if any(("memory_accesses" in data for data in measurementsdata)):
        for access_type in ("read", "write"):
            paths = {}

            memory_access_plot_path = (
                imgdir / f"memory_{access_type}s_plot_comparison"
            )

            render_multiple_time_series_plot(
                ydatas=[
                    [m["memory_accesses"]["read"] for m in measurementsdata]
                ],
                xdatas=[[m["profiler_timestamps"] for m in measurementsdata]],
                title="Memory reads comparison" if draw_titles else None,
                subtitles=None,
                xtitles=["Interval timestamp"],
                xunits=["s"],
                ytitles=["Memory reads"],
                yunits=["1/s"],
                legend_labels=[m["model_name"] for m in measurementsdata],
                outpath=memory_access_plot_path,
                outputext=image_formats,
                **SERVIS_PLOT_OPTIONS,
            )
            paths["persec"] = get_plot_wildcard_path(
                memory_access_plot_path, root_dir
            )

            cum_memory_access_plot_path = (
                imgdir / f"cumulative_memory_{access_type}s_plot_comparison"
            )

            LinePlot(
                title=f"Memory {access_type}s" if draw_titles else None,
                x_label="Interval timestamp",
                x_unit="s",
                y_label=f"Total memory {access_type}s",
                lines=[
                    (
                        m["profiler_timestamps"],
                        np.cumsum(m["memory_accesses"][access_type]),
                    )
                    for m in measurementsdata
                ],
                lines_labels=[m["model_name"] for m in measurementsdata],
                colors=SERVIS_PLOT_OPTIONS["colormap"],
            ).plot(cum_memory_access_plot_path, image_formats)
            paths["cumulative"] = get_plot_wildcard_path(
                cum_memory_access_plot_path, root_dir
            )

            if "memoryaccessesplotpath" not in report_variables:
                report_variables["memoryaccessesplotpath"] = {}

            report_variables["memoryaccessesplotpath"][access_type] = paths

    # peripheral accesses plot
    report_variables["peripheralaccessesplotpath"] = {}

    all_peripherals = set()

    for data in measurementsdata:
        all_peripherals = all_peripherals.union(
            data["peripheral_accesses"].keys()
        )

    for peripheral in all_peripherals:
        paths = defaultdict(dict)

        for access_type in ("read", "write"):
            xdata, ydata, labels = retrieve_non_zero_profiler_data(
                measurementsdata,
                ["peripheral_accesses", peripheral, access_type],
            )

            if not len(ydata):
                continue

            peripheral_access_plot_path = (
                imgdir / f"{peripheral}_{access_type}s_plot_comparison"
            )

            render_multiple_time_series_plot(
                ydatas=[ydata],
                xdatas=[xdata],
                title=f"{peripheral} reads comparison"
                if draw_titles
                else None,
                subtitles=None,
                xtitles=["Interval timestamp"],
                xunits=["s"],
                ytitles=[f"{peripheral} {access_type}s"],
                yunits=["1/s"],
                legend_labels=labels,
                outpath=peripheral_access_plot_path,
                outputext=image_formats,
                **SERVIS_PLOT_OPTIONS,
            )
            paths[access_type]["persec"] = get_plot_wildcard_path(
                peripheral_access_plot_path, root_dir
            )

            cum_peripheral_access_plot_path = (
                imgdir
                / f"cumulative_{peripheral}_{access_type}s_plot_comparison"
            )

            LinePlot(
                title=f"{peripheral} {access_type}s" if draw_titles else None,
                x_label="Interval timestamp",
                x_unit="s",
                y_label=f"Total {peripheral} {access_type}s",
                lines=[(x, np.cumsum(y)) for x, y in zip(xdata, ydata)],
                lines_labels=labels,
                colors=SERVIS_PLOT_OPTIONS["colormap"],
            ).plot(cum_peripheral_access_plot_path, image_formats)
            paths[access_type]["cumulative"] = get_plot_wildcard_path(
                cum_peripheral_access_plot_path, root_dir
            )

        if len(paths):
            report_variables["peripheralaccessesplotpath"][peripheral] = paths

    # exceptions plot
    xdata, ydata, labels = retrieve_non_zero_profiler_data(
        measurementsdata, ["exceptions"]
    )

    if len(ydata):
        paths = {}

        exceptions_plot_path = imgdir / "exceptions_plot_comparison"

        render_multiple_time_series_plot(
            ydatas=[ydata],
            xdatas=[xdata],
            title="Exceptions comparison" if draw_titles else None,
            subtitles=None,
            xtitles=["Interval timestamp"],
            xunits=["s"],
            ytitles=["Exceptions count"],
            yunits="1/s",
            legend_labels=labels,
            outpath=exceptions_plot_path,
            outputext=image_formats,
            **SERVIS_PLOT_OPTIONS,
        )
        paths["persec"] = get_plot_wildcard_path(
            exceptions_plot_path, root_dir
        )

        cum_exceptions_plot_path = (
            imgdir / "cumulative_exceptions_plot_comparison"
        )

        LinePlot(
            title="Total exceptions" if draw_titles else None,
            x_label="Interval timestamp",
            x_unit="s",
            y_label="Total exceptions",
            lines=[
                (
                    m["profiler_timestamps"],
                    np.cumsum(m["exceptions"]),
                )
                for m in measurementsdata
            ],
            colors=SERVIS_PLOT_OPTIONS["colormap"],
        ).plot(cum_exceptions_plot_path, image_formats)

        paths["cumulative"] = get_plot_wildcard_path(
            cum_exceptions_plot_path, root_dir
        )

        report_variables["exceptionsplotpath"] = paths

    with path(reports, "renode_stats_comparison.md") as reporttemplate:
        return create_report_from_measurements(
            reporttemplate, report_variables
        )


def generate_report(
    report_name: str,
    data: List[Dict],
    outputpath: Path,
    imgdir: Path,
    report_types: List[str],
    root_dir: Path,
    image_formats: Set[str],
    command: List[str] = [],
    cmap: Optional[Any] = None,
    colors: Optional[List] = None,
    draw_titles: bool = True,
    smaller_header: bool = False,
):
    """
    Generates an MyST report based on Measurements data.

    The report is saved to the file in ``outputpath``.

    Parameters
    ----------
    report_name : str
        Name for the report.
    data : List[Dict]
        Data for each model coming from the Measurements object,
        loaded i.e. from JSON files.
    outputpath : Path
        Path to the MyST file where the report will be saved.
    imgdir : Path
        Path to the directory where the report plots should be stored.
    report_types : List[str]
        List of report types that define the project, i.e.
        performance, classification.
    root_dir : Path
        When the report is a part of a larger MyST document (i.e. Sphinx docs),
        the `root_dir` parameter defines root directory of the document.
        It is used to compute relative paths in the document's references.
    image_formats : Set[str]
        Iterable object with extensions, in which images should be generated.
    command : List[str]
        Full command used to render this report, split into separate lines.
    cmap : Optional[Any]
        Color map to be used in the plots.
    colors : Optional[List]
        Colors to be used in the plots.
    draw_titles : bool
        Should titles be drawn on the plot.
    smaller_header : bool
        Use H2 header instead of H1.
    """
    from kenning.core.report import create_report_from_measurements

    reptypes = {
        PERFORMANCE: performance_report,
        CLASSIFICATION: classification_report,
        DETECTION: detection_report,
        RENODE: renode_stats_report,
    }
    comparereptypes = {
        PERFORMANCE: comparison_performance_report,
        CLASSIFICATION: comparison_classification_report,
        DETECTION: comparison_detection_report,
        RENODE: comparison_renode_stats_report,
    }

    header_data = {
        "report_name": report_name,
        "model_names": [],
        "command": [],
        "smaller_header": smaller_header,
    }

    for model_data in data:
        header_data["model_names"].append(model_data["model_name"])
        if "command" in model_data:
            header_data["command"] += model_data["command"] + [""]
        header_data[model_data["model_name"]] = model_data

    # add command only if previous one is not the same
    # if any(c1 != c2 for c1, c2 in zip(header_data['command'], command)):
    if header_data["command"] == command:
        header_data["command"] += command

    with path(reports, "header.md") as reporttemplate:
        content = create_report_from_measurements(reporttemplate, header_data)

    for typ in report_types:
        for i, model_data in enumerate(data):
            if len(data) > 1:
                imgprefix = model_data["model_name"] + "_"
            else:
                imgprefix = ""
            content += reptypes[typ](
                model_data,
                imgdir,
                imgprefix,
                root_dir,
                image_formats,
                color_offset=i,
                cmap=cmap,
                colors=colors,
                draw_titles=draw_titles,
            )
        if len(data) > 1:
            content += comparereptypes[typ](
                data,
                imgdir,
                root_dir,
                image_formats,
                cmap=cmap,
                colors=colors,
                draw_titles=draw_titles,
            )

    content = re.sub(r"[ \t]+$", "", content, 0, re.M)

    with open(outputpath, "w") as out:
        out.write(content)


def deduce_report_types(measurements_data: List[Dict]) -> List[str]:
    """
    Deduces what type of report should be generated based on measurements data.

    Report type is chosen only when all measurements data are compatible
    with it.

    Parameters
    ----------
    measurements_data : List[Dict]
        List with measurements data from which the report will be generated.

    Returns
    -------
    List[str]
        List with types of report
    """
    report_types = []

    def _append_type_if(_type: str, func: Callable):
        if all(map(func, measurements_data)):
            report_types.append(_type)

    _append_type_if(
        PERFORMANCE,
        lambda data: "target_inference_step" in data
        or "protocol_inference_step" in data,
    )
    _append_type_if(
        CLASSIFICATION, lambda data: "eval_confusion_matrix" in data
    )
    _append_type_if(DETECTION, lambda data: "eval_gtcount" in data)
    _append_type_if(RENODE, lambda data: "opcode_counters" in data)

    if len(report_types) == 0:
        KLogger.error(
            "There is no report type which is suitable for all measurements"
        )
        return []

    KLogger.info(f"Following report types were deduced: {report_types}")
    return report_types


def deduce_report_name(
    measurements_data: List[Dict], report_types: List[str]
) -> str:
    """
    Deduces simple report name based on measurements and its type.

    Parameters
    ----------
    measurements_data : List[Dict]
        List with measurements data from which the report will be generated.
    report_types : List[str]
        List with types of report.

    Returns
    -------
    str
        Report name
    """
    if len(measurements_data) > 1:
        report_name = (
            "Comparison of "
            f"{', '.join([d['model_name'] for d in measurements_data[:-1]])}"
            f" and {measurements_data[-1]['model_name']}"
        )
    elif "report_name" in measurements_data[0]:
        report_name = measurements_data[0]["report_name"]
    elif len(report_types) > 1:
        report_name = (
            f"{', '.join(report_types[:-1])} and "
            f"{report_types[-1]} of {measurements_data[0]['model_name']}"
        )
    else:
        report_name = (
            f"{report_types[0]} of " f"{measurements_data[0]['model_name']}"
        )
    report_name = report_name[0].upper() + report_name[1:]

    KLogger.info(f"Report name: {report_name}")
    return report_name


def generate_html_report(
    report_path: Path,
    output_folder: Path,
    debug: bool = False,
):
    """
    Runs Sphinx with HTML builder for generated report.

    Parameters
    ----------
    report_path : Path
        Path to the generated report file
    output_folder : Path
        Where generated HTML report should be saved
    debug : bool
        Debug mode -- allows to print more information
    """
    from sphinx.application import Sphinx
    from sphinx.cmd.build import handle_exception
    from sphinx.util.docutils import docutils_namespace, patch_docutils

    with path(reports, "conf.py") as _conf:
        override_conf = {
            # Include only report file
            "include_patterns": [f"{report_path.name}"],
            # Ensure report file isn't excluded
            "exclude_patterns": [],
            # Use report file as main source
            "master_doc": f'{report_path.with_suffix("").name}',
            # Static files for HTML
            "html_static_path": [f'{_conf.parent / "_static"}'],
            # Remove PFD button
            "html_theme_options.pdf_url": [],
            # Warning about using h2 header
            "suppress_warnings": ["myst.header"],
        }
        app = None
        try:
            with patch_docutils(_conf.parent), docutils_namespace():
                app = Sphinx(
                    report_path.parent,
                    _conf.parent,
                    output_folder,
                    output_folder / ".doctrees",
                    "html",
                    override_conf,
                    freshenv=False,
                )
                app.build(False, [str(report_path)])
        except Exception as ex:
            mock_args = namedtuple(
                "MockArgs", ("pdb", "verbosity", "traceback")
            )(pdb=debug, verbosity=debug, traceback=debug)
            handle_exception(app, mock_args, ex)
            KLogger.error(
                "Error occurred, HTML report won't be generated",
                ex.args,
                stack_info=True,
            )


class RenderReport(CommandTemplate):
    """
    Command-line template for rendering reports.
    """

    parse_all = True
    description = __doc__.split("\n\n")[0]

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(RenderReport, RenderReport).configure_parser(
            parser, command, types, groups
        )

        other_group = groups[DEFAULT_GROUP]
        # Group specific for this scenario,
        # doesn't have to be added to global groups
        report_group = parser.add_argument_group(GROUP_SCHEMA.format(REPORT))
        run_in_sequence = TEST in types

        other_group.add_argument(
            "--measurements",
            help="Path to the JSON files with measurements"
            + (
                f" created with {TEST} subcommand"
                if run_in_sequence
                else ". If more than one file is provided, model comparison will be generated."  # noqa: E501
            )
            + " It can be skipped when '--to-html' used, then HTML report will be rendered from previously generated report from '--report-path'",  # noqa: E501
            type=Path,
            nargs=1 if run_in_sequence else "*",
            default=[None],
            required=run_in_sequence,
        ).completer = FilesCompleter("*.json")
        report_group.add_argument(
            "--report-name",
            help="Name of the report",
            type=str,
        )
        other_group.add_argument(
            "--report-path",
            help="Path to the output MyST file",
            type=Path,
            required=True,
        )
        other_group.add_argument(
            "--to-html",
            help="Generate HTML version of the report, it can receive path to the folder where HTML will be saved",  # noqa: E501
            nargs="?",
            default=False,
            const=True,
            type=Path,
        ).completer = DirectoriesCompleter()
        report_group.add_argument(
            "--root-dir",
            help="Path to root directory for documentation (paths in the MyST file are relative to this directory)",  # noqa: E501
            type=Path,
        )
        report_group.add_argument(
            "--report-types",
            help="List of types that implement this report",
            nargs="+",
            choices=REPORT_TYPES,
        )
        report_group.add_argument(
            "--img-dir",
            help="Path to the directory where images will be stored",
            type=Path,
        )
        report_group.add_argument(
            "--model-names",
            help="Names of the models used to create measurements in order",
            nargs="+",
            type=str,
        )
        report_group.add_argument(
            "--only-png-images",
            help="Forcing to generate images only in PNG format, if not specified also images in HTML will be generated",  # noqa: E501
            action="store_true",
        )
        report_group.add_argument(
            "--skip-unoptimized-model",
            help="Do not use measurements of unoptimized model",
            action="store_true",
        )
        report_group.add_argument(
            "--smaller-header",
            help="Use smaller size for header containing report name",
            action="store_true",
        )
        return parser, groups

    @staticmethod
    def run(args, **kwargs):
        from kenning.core.drawing import (
            IMMATERIAL_COLORS,
            RED_GREEN_CMAP,
            SERVIS_PLOT_OPTIONS,
            choose_theme,
        )

        command = get_command()

        if args.to_html:
            if not isinstance(args.to_html, (str, Path)):
                args.to_html = Path(args.report_path).with_suffix("")
            if not args.measurements and args.report_path.exists():
                # Only render HTML report
                generate_html_report(
                    args.report_path, args.to_html, args.verbosity == "DEBUG"
                )
                return
            elif not args.measurements:
                raise argparse.ArgumentError(
                    None,
                    "HTML report cannot be generated, file from "
                    "'--report-path' does not exist. Please, make sure the "
                    "path is correct or use '--measurements' to generate new "
                    "report.",
                )

        if not args.measurements:
            raise argparse.ArgumentError(
                None,
                "'--measurements' have to be defined to generate new report. "
                "If only HTML version from existing report has to be "
                "rendered, please use '--to-html' flag",
            )

        root_dir = args.root_dir
        if root_dir is None:
            root_dir = args.report_path.parent.absolute()

        if not args.img_dir:
            img_dir = root_dir / "img"
        else:
            img_dir = args.img_dir
        img_dir.mkdir(parents=True, exist_ok=True)

        if args.model_names is not None and len(args.measurements) != len(
            args.model_names
        ):
            KLogger.warning(
                "Number of model names differ from number of measurements! "
                "Ignoring --model-names argument"
            )
            args.model_names = None

        image_formats = {"png"}
        if not args.only_png_images:
            image_formats |= {"html"}

        measurementsdata = []
        for i, measurementspath in enumerate(args.measurements):
            with open(measurementspath, "r") as measurementsfile:
                measurements = json.load(measurementsfile)
            if args.model_names is not None:
                measurements["model_name"] = args.model_names[i]
            elif "model_name" not in measurements:
                measurements["model_name"] = get_model_name(measurementspath)
            measurements["model_name"] = measurements["model_name"].replace(
                " ", "_"
            )
            # Append measurements of unoptimized data separately
            if (
                not args.skip_unoptimized_model
                and UNOPTIMIZED_MEASUREMENTS in measurements
            ):
                unoptimized = measurements[UNOPTIMIZED_MEASUREMENTS]
                del measurements[UNOPTIMIZED_MEASUREMENTS]
                if "model_name" not in unoptimized:
                    unoptimized[
                        "model_name"
                    ] = f"unoptimized_{measurements['model_name']}"
                measurementsdata.append(unoptimized)
            measurementsdata.append(measurements)

        report_types = args.report_types
        if not report_types:
            report_types = deduce_report_types(measurementsdata)
        if report_types is None:
            raise argparse.ArgumentError(
                None,
                "Report types cannot be deduced. Please specify "
                "'--report-types' or make sure the path is correct "
                "measurements were chosen.",
            )

        report_name = args.report_name
        if report_name is None:
            report_name = deduce_report_name(measurementsdata, report_types)
        for measurements in measurementsdata:
            if "build_cfg" in measurements:
                measurements["build_cfg"] = json.dumps(
                    measurements["build_cfg"], indent=4
                ).split("\n")

            if "report_name" not in measurements:
                measurements["report_name"] = deduce_report_name(
                    [measurements], report_types
                )
            measurements["report_name_simple"] = re.sub(
                r"[\W]",
                "",
                measurements["report_name"].lower().replace(" ", "_"),
            )

        SERVIS_PLOT_OPTIONS["colormap"] = IMMATERIAL_COLORS
        cmap = RED_GREEN_CMAP
        colors = IMMATERIAL_COLORS

        with choose_theme(
            custom_bokeh_theme=True,
            custom_matplotlib_theme=True,
        ):
            generate_report(
                report_name,
                measurementsdata,
                args.report_path,
                img_dir,
                report_types,
                root_dir,
                image_formats,
                command,
                cmap=cmap,
                colors=colors,
                draw_titles=False,
                smaller_header=args.smaller_header,
            )

        if args.to_html:
            generate_html_report(
                args.report_path, args.to_html, args.verbosity == "DEBUG"
            )


if __name__ == "__main__":
    sys.exit(RenderReport.scenario_run())
