# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script for running Kenning Flows.
"""

import argparse
import json
import sys
from typing import List, Optional, Tuple

from argcomplete.completers import FilesCompleter

from kenning.cli.command_template import (
    FLOW,
    GROUP_SCHEMA,
    ArgumentsGroups,
    CommandTemplate,
)
from kenning.core.flow import KenningFlow
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import ResourceURI


class FlowRunner(CommandTemplate):
    """
    Command template for running Kenning applications.
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
        parser, groups = super(FlowRunner, FlowRunner).configure_parser(
            parser, command, types, groups
        )

        flow_group = parser.add_argument_group(GROUP_SCHEMA.format(FLOW))

        flow_group.add_argument(
            "--json-cfg",
            help="The path to the input JSON file with configuration of the graph",  # noqa: E501
            type=ResourceURI,
            required=True,
        ).completer = FilesCompleter("*.json")
        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, **kwargs):
        KLogger.set_verbosity(args.verbosity)

        with open(args.json_cfg, "r") as f:
            json_cfg = json.load(f)

        flow: KenningFlow = KenningFlow.from_json(json_cfg)
        _ = flow.run()

        KLogger.info("Processing has finished")
        return 0


if __name__ == "__main__":
    sys.exit(FlowRunner.scenario_run())
