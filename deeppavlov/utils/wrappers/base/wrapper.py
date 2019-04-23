import yaml
import json
from abc import ABCMeta, abstractmethod
from typing import Hashable, DefaultDict, List, Optional, Union
from pathlib import Path


class ConfigBase(metaclass=ABCMeta):
    processor_type: str
    infer_batch_size: int
    infer_args_names: List[str]
    next_arg_request_template: str
    response_template: str

    # host: Optional[str]
    # port: Optional[Union[str, int]]
    # https: Optional[bool]
    # https_cert_path: Optional[Path]
    # https_key_path: Optional[Path]

    def __init__(self, config_file: Optional[Path] = None, **kwargs):
        if config_file:
            with config_file.open('r') as f:
                if config_file.suffix in ['.json']:
                    config_dict = json.load(f)
                elif config_file.suffix in ['.yaml', '.yml']:
                    config_dict = yaml.load(f)
                else:
                    raise ValueError(f'Config file type should be in [.json, .yaml, .yaml]')
        else:
            config_dict = {**kwargs}


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
    _response_template: str

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
        self._response_template = config.response_template

    async def process_utterance(self, utterance: str, utterance_id: Hashable) -> None:
        await self._processor(utterance, utterance_id)

    async def _process_stateless(self, utterance: str, utterance_id: Hashable):
        if not self._states[utterance_id]:
            self._states[utterance_id] = {'expected_args': self._infer_args_names, 'received_values': []}

        if utterance:
            self._states[utterance_id].pop(0)
            self._states[utterance_id]['received_values'].append(utterance)

        if self._states[utterance_id]['expected_args']:
            response = self._next_arg_request_template.format(self._states[utterance_id]['expected_args'][0])
        else:
            infer_args = self._states[utterance_id]['expected_args']
            response = await self._infer_gateway.infer(*infer_args)

            if not isinstance(response, tuple):
                response = (str(response), )

            response = self._response_template.format(*response)
            self._states[utterance_id] = None

        await self._out_gateway.respond(response, utterance_id)

    async def _process_skill(self, utterance: str, utterance_id: Hashable):
        self._histories[utterance_id].append(utterance)

        response = await self._infer_gateway.infer(utterance, self._histories[utterance_id], self._states[utterance_id])
        # TODO: may be this unpacking logic should be moved to infer gateway
        response = list(response) + [None] * (2 - len(response)) if isinstance(response, tuple) else [response, None]

        self._histories[utterance_id].append(str(response[0]))
        self._states[utterance_id] = response[1]

        await self._out_gateway.respond(str(response[0]), utterance_id)

    async def _process_agent(self, utterance: str, utterance_id: Hashable):
        response = await self._infer_gateway.infer(utterance, utterance_id)
        await self._out_gateway.respond(response, utterance_id)
