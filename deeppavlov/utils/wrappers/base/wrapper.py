import re
import yaml
import json
from abc import ABCMeta, abstractmethod
from typing import Hashable, DefaultDict, List, Optional, Union
from pathlib import Path


def safe_format(string: str, *args):
    format_values = [str(arg) for arg in args]
    format_placeholders = re.findall(r'\{\}', string)
    delta = len(format_placeholders) - len(format_values)

    if delta > 0:
        format_values.extend([''] * delta)

    result = string if len(format_values) == 0 else string.format(*format_values)

    return result


class ConfigBase(metaclass=ABCMeta):
    processor_type: str
    infer_batch_size: int
    infer_args_names: List[str]
    next_arg_request_template: str
    stateless_response_template: str

    def __init__(self, config_file: Optional[Path] = None, **kwargs) -> None:
        # default values
        self.processor_type = 'stateless'
        self.infer_batch_size = 10
        self.infer_args_names = ['context']
        self.next_arg_request_template = 'Please, enter {} argument'
        self.stateless_response_template = '{}'

        if config_file:
            with config_file.open('r') as f:
                if config_file.suffix in ['.json']:
                    config_dict = json.load(f)
                elif config_file.suffix in ['.yaml', '.yml']:
                    config_dict = yaml.load(f)
                else:
                    raise ValueError(f'Config file type should be in [.json, .yaml, .yaml]')

                if not isinstance(config_dict, dict):
                    raise ValueError('Config file should have one level dict-like structure')

        else:
            config_dict = {**kwargs}

        for attr_name, attr_value in config_dict.items():
            setattr(self, attr_name, attr_value)


class InGatewayBase(metaclass=ABCMeta):
    _utterance_processor: callable

    def __init__(self, utterance_processor: callable) -> None:
        self._utterance_processor = utterance_processor

    async def _process_utterance(self, utterance: str, utterance_id: Hashable) -> None:
        await self._utterance_processor(utterance, utterance_id)


class OutGatewayBase(metaclass=ABCMeta):
    @abstractmethod
    async def respond(self, utterance: str, utterance_id: Hashable) -> None:
        pass


class InferGatewayBase(metaclass=ABCMeta):
    @abstractmethod
    async def infer(self, *args, **kwargs) -> Union[str, tuple]:
        pass


class Wrapper(metaclass=ABCMeta):
    _in_gateway: InGatewayBase
    _out_gateway: OutGatewayBase
    _infer_gateway: InferGatewayBase

    _states: dict
    _histories: DefaultDict[list]

    _processor: callable

    _infer_batch_size: int

    _infer_args_names: List[str]
    _next_arg_request_template: str
    _stateless_response_template: str

    def __init__(self, config: ConfigBase) -> None:
        processors = {
            'stateless': self._process_stateless,
            'skill': self._process_skill,
            'agent': self._process_agent
        }

        self._processor = processors[config.processor_type]
        self._infer_batch_size = config.infer_batch_size
        self._infer_args_names = config.infer_args_names
        self._next_arg_request_template = config.next_arg_request_template
        self._stateless_response_template = config.stateless_response_template

    async def process_utterance(self, utterance: str, utterance_id: Hashable) -> None:
        await self._processor(utterance, utterance_id)

    async def _process_stateless(self, utterance: str, utterance_id: Hashable) -> None:
        if not self._states[utterance_id]:
            self._states[utterance_id] = {'expected_args': self._infer_args_names, 'received_values': []}

        if utterance:
            self._states[utterance_id].pop(0)
            self._states[utterance_id]['received_values'].append(utterance)

        if self._states[utterance_id]['expected_args']:
            response = safe_format(self._next_arg_request_template, self._states[utterance_id]['expected_args'][0])
        else:
            infer_args = self._states[utterance_id]['expected_args']
            response = await self._infer_gateway.infer(*infer_args)

            if not isinstance(response, tuple):
                response = (str(response), )

            response = safe_format(self._stateless_response_template, *response)

            self._states[utterance_id] = None

        await self._out_gateway.respond(response, utterance_id)

    async def _process_skill(self, utterance: str, utterance_id: Hashable) -> None:
        self._histories[utterance_id].append(utterance)

        response = await self._infer_gateway.infer(utterance, self._histories[utterance_id], self._states[utterance_id])
        # TODO: may be this unpacking logic should be moved to infer gateway
        response = list(response) + [None] * (2 - len(response)) if isinstance(response, tuple) else [response, None]

        self._histories[utterance_id].append(str(response[0]))
        self._states[utterance_id] = response[1]

        await self._out_gateway.respond(str(response[0]), utterance_id)

    async def _process_agent(self, utterance: str, utterance_id: Hashable) -> None:
        response = await self._infer_gateway.infer(utterance, utterance_id)
        await self._out_gateway.respond(response, utterance_id)
