# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for preparing and serializing class arguments.
"""
import json
import os.path

import jsonschema
import argparse
from abc import ABC
from typing import Dict, Tuple
from pathlib import Path
import numpy as np

from kenning.utils.resource_manager import ResourceURI

"""
arguments_structure is a mapping (argument_name -> keywords)

Supported keywords:
argparse_name: Name that is prompted as an argparse argument.
    If argparse_name is not specified then JSON uses argument_name
    and argparse adds -- prefix and change all underscores to hyphens.
    E.g. model_path -> --model-path.

    If it is specified then argparse uses it and JSON uses argument with
    hyphens stripped from the prefix and changed to underscores.
    E.g. --some-name -> some_name.
description: Description of the argument.
type: Same as 'type' in argparse. The argument is converted to this value.
    Possible values for type: [int, float, str, bool, Path].

    Note that for bool types, the argument has to declare its default value.
default: Default value of the argument.
required: Determines whether the argument is required.
enum: List of possible values of the argument.
is_list: Determines whether the argument is a list of arguments.
    Possible values for is_list: [True, False].
    By default it is False.
    List of bool arguments is not supported
nullable: Determines whether the argument can be None.
    Possible values for nullable: [True, False].
    By default it is False.
    It is used only for JSON.
    Note that for a property to be nullable it has to be either a list
    or have a type defined.

Examples:

'compiled_model_path': {
    'argparse_name': '--model-path',
    'description': 'The path to the compiled model output',
    'type': Path,
    'required': True,
    'enum': ['/home/Documents', '/tmp']
}

'inputdims': {
    'description': 'Dimensionality of the inputs',
    'type': int,
    'default': [224, 224, 3],
    'is_list': True,
    'nullable': True
}
"""

supported_keywords = [
    'argparse_name',
    'description',
    'type',
    'default',
    'required',
    'enum',
    'is_list',
    'nullable'
]


def from_argparse_name(s):
    return s.lstrip('-').replace('-', '_')


def to_argparse_name(s):
    return '--' + s.replace('_', '-')


def convert_to_jsontype(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


type_to_jsontype = {
    Path: 'string',
    ResourceURI: 'string',
    str: 'string',
    float: 'number',
    int: 'integer',
    bool: 'boolean',
    object: 'object',
}

jsontype_to_type = {
    'string': str,
    'number': float,
    'integer': int,
    'boolean': bool
}


def serialize(obj: object, normalize: bool = True) -> Dict:
    """
    Serializes the given object into a dictionary.

    It serializes all variables mentioned in the dictionary
    returned by a `form_parameterschema` function.

    Note that object has to implement `form_parameterschema`
    which uses `add_parameterschema_argument` function
    to be serializable.

    Parameters
    ----------
    obj : object
        Object to serialize.
    normalize : bool
        Determines whether to convert value to JSON type.

    Returns
    -------
    str :
        Serialized object.
    """
    if hasattr(obj, 'form_parameterschema'):
        properties = obj.form_parameterschema()['properties']
        to_serialize = {
            keywords['real_name']: name
            for name, keywords in properties.items()
        }
    else:
        return {}

    serialized_dict = {}

    for name, value in vars(obj).items():
        if name in to_serialize:
            serialized_dict[to_serialize[name]] = (
                convert_to_jsontype(value) if normalize else value
            )

    return serialized_dict


def serialize_inference(
        dataset,
        model,
        optimizers,
        runtimeprotocol,
        runtime) -> Dict:
    """
    Serializes the given objects into a dictionary which
    is a valid input for `inference_tester.py`.

    Parameters
    ----------
    dataset : Dataset
        Dataset to serialize.
    model : ModelWrapper
        ModelWrapper to serialize.
    optimizers : Union[List[Optimizer], Optimizer]
        Optimizers to serialize.
    runtimeprotocol : RuntimeProtocol
        RuntimeProtocol to serialize.
    runtime : Runtime
        Runtime to serialize.

    Returns
    -------
    Dict :
        Serialized inference.
    """
    def object_to_module(obj):
        return type(obj).__module__ + '.' + type(obj).__name__

    serialized_dict = {}

    for obj, name in zip(
        [dataset, model, runtimeprotocol, runtime],
        ['dataset', 'model_wrapper', 'runtime_protocol', 'runtime']
    ):
        if obj:
            serialized_dict[name] = {}
            serialized_dict[name]['type'] = object_to_module(obj)

            serialized_obj = serialize(obj)
            serialized_dict[name]['parameters'] = serialized_obj

    if optimizers:
        if type(optimizers) is not list:
            optimizers = [optimizers]

        serialized_dict['optimizers'] = []
        for optimizer in optimizers:
            optimizer_dict = {}
            optimizer_dict['type'] = object_to_module(optimizer)

            serialized_obj = serialize(optimizer)
            optimizer_dict['parameters'] = serialized_obj

            serialized_dict['optimizers'].append(optimizer_dict)

    return serialized_dict


def get_parsed_json_dict(schema, json_dict: Dict) -> Dict:
    """
    Validates the given dictionary with the schema.
    Then it adds default values for missing
    arguments and converts the arguments to the appropriate types
    as jsonschema can not do that. Finally it provides new names
    that match the constructor arguments.

    Parameters
    ----------
    schema : Dict
        Schema to validate with.
    json_dict : Dict
        Dictionary to validate.

    Returns
    -------
    Dict :
        Validated dictionary with arguments.
    """
    jsonschema.validate(
        instance=json_dict,
        schema=schema
    )

    converted_json_dict = {}

    # Including default values
    for name, keywords in schema['properties'].items():
        if name not in json_dict and 'default' in keywords:
            converted_json_dict[name] = keywords['default']
        elif name in json_dict:
            converted_json_dict[name] = json_dict[name]

    # Converting arguments to the final types.
    # Following argparse convention, if our value is None
    # then we do not convert it.
    for name, value in converted_json_dict.items():
        keywords = schema['properties'][name]

        if 'convert-type' not in keywords or not value:
            converted_json_dict[name] = value
            continue

        # do not convert to an object as this should be done by the class
        # which called this function
        if keywords['convert-type'] == object:
            converted_json_dict[name] = value
            continue

        converter = keywords['convert-type']

        if 'type' in keywords and 'array' in keywords['type']:
            converted_json_dict[name] = [
                converter(v) if v else v for v in value
            ]
        else:
            converted_json_dict[name] = converter(value)

    # Changing names so they match the constructor arguments
    converted_json_dict = {
        schema['properties'][name]['real_name']: value
        for name, value in converted_json_dict.items()
    }

    return converted_json_dict


def get_parsed_args_dict(cls: type, args: argparse.Namespace) -> Dict:
    """
    Converts namespace provided by arguments parser into dictionary.

    Parameters
    ----------
    cls : type
        Class of object being parsed.
    args : argparse.Namespace
        Namespace provided by arguments parser.

    Returns
    -------
    Dict :
        Dictionary with arguments.
    """
    # retrieve all arguments from arguments_structure of this class and all of
    # its parent classes
    classes = [cls]
    args_structure = {}
    while len(classes):
        curr_cls = classes.pop(0)
        classes.extend(curr_cls.__bases__)
        if not hasattr(curr_cls, 'arguments_structure'):
            continue
        args_structure = dict(
            args_structure,
            **curr_cls.arguments_structure
        )

    # parse arguments
    parsed_args = {}
    for arg_name, arg_properties in args_structure.items():
        if 'argparse_name' in arg_properties:
            argparse_name = from_argparse_name(arg_properties['argparse_name'])
        else:
            argparse_name = arg_name

        if hasattr(args, argparse_name):
            value = getattr(args, argparse_name)
        else:
            try:
                value = arg_properties['default']
            except KeyError:
                raise Exception(
                    f'No default value provided for {argparse_name}'
                )

        # For arguments of type 'object' value is embedded in a JSON file
        if 'type' in arg_properties and arg_properties['type'] == object:
            if value is None:
                try:
                    value = arg_properties['default']
                    parsed_args[arg_name] = value
                    continue
                except KeyError:
                    raise Exception(
                        f'No default value provided for {argparse_name}'
                    )

            if not os.path.exists(value):
                raise Exception(
                    f'JSON configuration file {value} doesnt exist'
                )

            with open(value, 'r') as file:
                value = json.load(file)
                parsed_args[arg_name] = value
                continue

        # convert type
        if 'type' in arg_properties and value is not None:
            value = arg_properties['type'](value)

        parsed_args[arg_name] = value

    return parsed_args


def add_argparse_argument(
        group: argparse._ArgumentGroup,
        struct: Dict[str, Dict],
        *names: str):
    """
    Adds arguments to the argparse group based on the given
    struct and names. If no names are given it uses all
    properties of the struct.

    Note that the function modifies the given group.

    If argument with name 'argparse_name' already exists in the
    group the existing argument is overridden with the new one.

    Parameters
    ----------
    group : argparse._ArgumentGroup
        Argparse group that is filled with new properties.
    struct : Dict[str, Dict]
        Struct with properties described.
    *names : str
        Names of the properties that are to be added to the group.
        If empty every property in struct is used.

    Raises
    ------
    KeyError :
        Raised if there is a keyword that is not recognized.
    """
    if not names:
        names = struct.keys()

    for name in names:
        prop = struct[name]

        for p in prop:
            if p not in supported_keywords:
                raise KeyError(f'{p} is not a supported keyword')

        if 'argparse_name' in prop:
            argparse_name = prop['argparse_name']
        else:
            argparse_name = to_argparse_name(name)

        keywords = {}
        if 'type' in prop:
            if prop['type'] is bool:
                assert 'default' in prop and prop['default'] in [True, False]

                keywords['action'] = (
                    'store_false' if prop['default']
                    else 'store_true'
                )
            else:
                keywords['type'] = prop['type']
        if 'description' in prop:
            keywords['help'] = prop['description']
        if 'default' in prop:
            keywords['default'] = prop['default']
        if 'required' in prop and prop['required']:
            keywords['required'] = prop['required']
        if 'enum' in prop:
            keywords['choices'] = prop['enum']
        if 'is_list' in prop and prop['is_list']:
            keywords['nargs'] = '+'

        group.add_argument(argparse_name, **keywords)


def add_parameterschema_argument(
        schema: Dict,
        struct: Dict[str, Dict],
        *names: str):
    """
    Adds arguments to the schema based on the given
    struct and names. If no names are given it uses all
    properties of the struct.

    Note that the function modifies the given schema.

    If argument with name 'argschema_name' already exists in the
    schema the existing argument is overridden with the new one.

    Parameters
    ----------
    schema : Dict
        Schema that is filled with new properties.
    struct : Dict[str, Dict]
        Struct with properties described.
    *names : str
        Names of the properties that are to be added to the group.
        If empty every property in struct is used.

    Raises
    ------
    KeyError :
        Raised if there is a keyword that is not recognized or if there is
        already a property with a different `argparse_name` and the same
        property name.
    """
    if 'properties' not in schema:
        schema['properties'] = {}

    if not names:
        names = struct.keys()

    for name in names:
        prop = struct[name]

        for p in prop:
            if p not in supported_keywords:
                raise KeyError(f'{p} is not a supported keyword')

        if 'argparse_name' in prop:
            argparse_name = prop['argparse_name']
            argschema_name = from_argparse_name(argparse_name)
        else:
            argschema_name = name

        # Check if there is a property that is not going to be overriden
        # by the new property but has the same property name
        for k, p in schema['properties'].items():
            if p['real_name'] == name and k != argschema_name:
                raise KeyError(f'{p} already has a property name: {name}')

        # If the property is going to be overriden it has to be removed
        # from the list of required properties
        try:
            schema['required'].remove(argschema_name)
        except (KeyError, ValueError):
            pass

        schema['properties'][argschema_name] = {}
        keywords = schema['properties'][argschema_name]
        keywords['real_name'] = name

        # Case for a list of keywords
        if 'is_list' in prop and prop['is_list']:
            keywords['type'] = ['array']

            if 'type' in prop:
                assert prop['type'] is not bool

                keywords['convert-type'] = prop['type']
                keywords['items'] = {
                    'type': type_to_jsontype[prop['type']]
                }
        # Case for a single argument
        else:
            if 'type' in prop:
                keywords['convert-type'] = prop['type']
                keywords['type'] = [type_to_jsontype[prop['type']]]

        if 'description' in prop:
            keywords['description'] = prop['description']
        if 'default' in prop:
            keywords['default'] = prop['default']
        if 'required' in prop and prop['required']:
            if 'required' not in schema:
                schema['required'] = []
            schema['required'].append(argschema_name)
        if 'enum' in prop:
            keywords['enum'] = prop['enum']
        if 'nullable' in prop and prop['nullable']:
            if 'type' in keywords:
                keywords['type'] += ['null']


class ArgumentsHandler(ABC):
    """
    Class responsible for creating parsers for arguments from command line or
    json configs.

    The child class should define its own `arguments_structure` and
    from_argparse/from_json methods so that it could be instantiated from
    command line arguments or json config.
    """

    arguments_structure = {}

    @classmethod
    def form_parameterschema(cls) -> Dict:
        """
        Creates parameter schema based on `arguments_structure` of class and
        its all parent classes.

        Returns
        -------
        Dict :
            Parameter schema for the class.
        """
        classes = [cls]
        parameterschema = {
            'type': 'object',
            'additionalProperties': False
        }

        while len(classes):
            curr_cls = classes.pop(0)
            classes.extend(curr_cls.__bases__)
            if not hasattr(curr_cls, 'arguments_structure'):
                continue
            add_parameterschema_argument(
                parameterschema,
                curr_cls.arguments_structure
            )

        return parameterschema

    @classmethod
    def form_argparse(
            cls) -> Tuple[argparse.ArgumentParser, argparse._ArgumentGroup]:
        """
        Creates argparse parser based on `arguments_structure` of class and its
        all parent classes.

        Returns
        -------
        Tuple[argparse.ArgumentParser, argparse._ArgumentGroup] :
            Tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer.
        """
        classes = [cls]
        parser = argparse.ArgumentParser(
            add_help=False,
            conflict_handler='resolve'
        )
        group = None

        while len(classes):
            curr_cls = classes.pop(0)
            classes.extend(curr_cls.__bases__)
            if not hasattr(curr_cls, 'arguments_structure'):
                continue
            group = parser.add_argument_group(
                title=f'{curr_cls.__name__} arguments'
            )
            add_argparse_argument(
                group,
                curr_cls.arguments_structure
            )

        return parser, group
