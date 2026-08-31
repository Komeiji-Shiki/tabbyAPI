import re
import json
from uuid import uuid4
from common.logger import xlogger
from endpoints.OAI.types.tools import ToolCall, Tool
from endpoints.OAI.utils.toolcall_formats.common import coerce_param_value

"""
Qwen3.5 / Qwen3-Coder - pseudo-XML syntax

Raw format:
    <tool_call>
        <function=__FUNCTION_NAME__>
            <parameter=__PARAMETER_NAME_1__>
                __PARAMETER_1__
            </parameter>
            <parameter=__PARAMETER_NAME_2__>
                __PARAMETER_2__
            </parameter>
            ...
        </function>
    </tool_call>
"""

# TODO: the outer <tool_call> wrapper is supposedly optional in some deployments; the parser
#   handles both, but detecting tool calls in the stream currently relies on <tool_call> being
#   emitted by the model.

TOOLCALL_START = "<tool_call>"
TOOLCALL_END = "</tool_call>"

_OUTER = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNC = re.compile(r"<function=([^>\s]+)[^>]*>(.*?)</function>", re.DOTALL)
_PARAM = re.compile(r"<parameter=([^>\s]+)[^>]*>(.*?)</parameter>", re.DOTALL)


class StreamToolCallParser:
    """Incrementally convert Qwen pseudo-XML tool calls into OAI deltas."""

    _FUNCTION_OPEN = re.compile(r"<function=([^>\s]+)[^>]*>")
    _PARAMETER_OPEN = re.compile(r"<parameter=([^>\s]+)[^>]*>")
    _FUNCTION_CLOSE = "</function>"
    _PARAMETER_CLOSE = "</parameter>"
    _TOOL_OPEN = "<tool_call>"
    _TOOL_CLOSE = "</tool_call>"

    def __init__(self, tools=None):
        self._buffer = ""
        self._state = "outside"
        self._current = None
        self._parameter_name = None
        self._parameter_value = ""
        self._parameter_mode = None
        self._parameter_leading = True
        self._parameter_trailing_ws = ""
        self._parameter_schemas = self._build_parameter_schemas(tools)
        self._next_index = 0
        self.completed_calls = 0

    @property
    def has_tool_calls(self) -> bool:
        """Whether at least one complete function call was emitted."""

        return self.completed_calls > 0

    def feed(self, text: str) -> list[dict]:
        """Consume arbitrary text chunks and return any new OAI tool deltas."""

        if text:
            self._buffer += text

        deltas = []
        while self._buffer:
            if self._state == "outside":
                if self._buffer.startswith(self._TOOL_OPEN):
                    self._buffer = self._buffer[len(self._TOOL_OPEN) :]
                    continue
                if self._buffer.startswith(self._TOOL_CLOSE):
                    self._buffer = self._buffer[len(self._TOOL_CLOSE) :]
                    continue

                function_match = self._FUNCTION_OPEN.search(self._buffer)
                if function_match:
                    self._buffer = self._buffer[function_match.end() :]
                    deltas.append(self._start_function(function_match.group(1)))
                    self._state = "function"
                    continue

                self._buffer = self._hold_partial(
                    self._buffer,
                    ("<function", "<tool_call", "</tool_call"),
                )
                break

            elif self._state == "function":
                parameter_match = self._PARAMETER_OPEN.search(self._buffer)
                function_end = self._buffer.find(self._FUNCTION_CLOSE)

                if function_end >= 0 and (
                    parameter_match is None or function_end < parameter_match.start()
                ):
                    self._buffer = self._buffer[function_end + len(self._FUNCTION_CLOSE) :]
                    deltas.extend(self._finish_function())
                    self._state = "outside"
                    continue

                if parameter_match:
                    self._buffer = self._buffer[parameter_match.end() :]
                    deltas.extend(self._start_parameter(parameter_match.group(1)))
                    self._state = "parameter"
                    continue

                self._buffer = self._hold_partial(
                    self._buffer,
                    ("<parameter", "</function"),
                )
                break

            else:  # parameter
                parameter_end = self._buffer.find(self._PARAMETER_CLOSE)
                if parameter_end < 0:
                    if self._parameter_mode is not None:
                        value, self._buffer = self._split_partial_suffix(
                            self._buffer,
                            self._PARAMETER_CLOSE,
                        )
                        deltas.extend(self._stream_parameter_value(value))
                    break

                value = self._buffer[:parameter_end]
                self._buffer = self._buffer[
                    parameter_end + len(self._PARAMETER_CLOSE) :
                ]
                if self._parameter_mode is None:
                    self._parameter_value += value
                    deltas.extend(self._finish_parameter())
                else:
                    deltas.extend(self._stream_parameter_value(value))
                    deltas.extend(self._finish_streaming_parameter())
                self._state = "function"

        return deltas

    def finish(self) -> list[dict]:
        """Flush complete buffered syntax without fabricating truncated calls."""

        return self.feed("")

    @staticmethod
    def _hold_partial(text: str, prefixes: tuple[str, ...]) -> str:
        """Keep a possible split tag while discarding unrelated text."""

        last_marker = max((text.rfind(prefix) for prefix in prefixes), default=-1)
        if last_marker >= 0:
            return text[last_marker:]

        for prefix in sorted(prefixes, key=len, reverse=True):
            limit = min(len(prefix) - 1, len(text))
            for length in range(limit, 0, -1):
                if text.endswith(prefix[:length]):
                    return text[-length:]
        return ""

    def _start_function(self, name: str) -> dict:
        call = {
            "index": self._next_index,
            "id": f"call_{uuid4().hex[:24]}",
            "type": "function",
            "name": name,
            "has_arguments": False,
        }
        self._next_index += 1
        self._current = call
        return {
            "index": call["index"],
            "id": call["id"],
            "type": call["type"],
            "function": {"name": call["name"], "arguments": ""},
        }

    @staticmethod
    def _build_parameter_schemas(tools) -> dict:
        schemas = {}
        for tool in tools or []:
            data = tool.model_dump() if hasattr(tool, "model_dump") else tool
            if not isinstance(data, dict):
                continue
            function = data.get("function") or data
            name = function.get("name")
            parameters = function.get("parameters") or {}
            properties = parameters.get("properties") or {}
            if name and isinstance(properties, dict):
                schemas[name] = properties
        return schemas

    @classmethod
    def _schema_type(cls, schema: dict) -> str | None:
        schema_type = schema.get("type")
        if isinstance(schema_type, str):
            return schema_type
        if isinstance(schema_type, list):
            non_null = [item for item in schema_type if item != "null"]
            if len(non_null) == 1:
                return non_null[0]

        for keyword in ("anyOf", "oneOf"):
            branches = schema.get(keyword)
            if not isinstance(branches, list):
                continue
            branch_types = {
                cls._schema_type(branch)
                for branch in branches
                if isinstance(branch, dict)
            }
            branch_types.discard(None)
            branch_types.discard("null")
            if len(branch_types) == 1:
                return branch_types.pop()

        if "properties" in schema:
            return "object"
        if "items" in schema:
            return "array"
        return None

    def _parameter_stream_mode(self, name: str) -> str | None:
        if self._current is None:
            return None

        schema = self._parameter_schemas.get(self._current["name"], {}).get(name, {})
        if not isinstance(schema, dict):
            return None

        schema_type = self._schema_type(schema)

        if schema_type == "string":
            return "string"
        if schema_type in ("object", "array"):
            return "raw"
        return None

    def _argument_prefix(self, name: str) -> str:
        separator = "{" if not self._current["has_arguments"] else ","
        self._current["has_arguments"] = True
        key = json.dumps(name, ensure_ascii=False)
        return f"{separator}{key}:"

    def _start_parameter(self, name: str) -> list[dict]:
        self._parameter_name = name.strip()
        self._parameter_value = ""
        self._parameter_mode = self._parameter_stream_mode(self._parameter_name)
        self._parameter_leading = True
        self._parameter_trailing_ws = ""

        if self._parameter_mode is None:
            return []

        arguments = self._argument_prefix(self._parameter_name)
        if self._parameter_mode == "string":
            arguments += '"'
        return [self._argument_delta(arguments)]

    @staticmethod
    def _split_partial_suffix(text: str, marker: str) -> tuple[str, str]:
        limit = min(len(marker) - 1, len(text))
        for length in range(limit, 0, -1):
            if text.endswith(marker[:length]):
                return text[:-length], text[-length:]
        return text, ""

    def _stream_parameter_value(self, value: str) -> list[dict]:
        if not value:
            return []

        if self._parameter_leading:
            value = value.lstrip()
            if not value:
                return []
            self._parameter_leading = False

        value = self._parameter_trailing_ws + value
        self._parameter_trailing_ws = ""
        stripped = value.rstrip()
        self._parameter_trailing_ws = value[len(stripped) :]
        if not stripped:
            return []

        if self._parameter_mode == "string":
            stripped = json.dumps(stripped, ensure_ascii=False)[1:-1]
        return [self._argument_delta(stripped)]

    def _finish_streaming_parameter(self) -> list[dict]:
        arguments = '"' if self._parameter_mode == "string" else ""
        self._reset_parameter()
        return [self._argument_delta(arguments)] if arguments else []

    def _argument_delta(self, arguments: str) -> dict:
        return {
            "index": self._current["index"],
            "function": {"arguments": arguments},
        }

    def _reset_parameter(self):
        self._parameter_name = None
        self._parameter_value = ""
        self._parameter_mode = None
        self._parameter_leading = True
        self._parameter_trailing_ws = ""

    def _finish_parameter(self) -> list[dict]:
        if self._current is None:
            return []

        value = coerce_param_value(self._parameter_value.strip())
        encoded_value = json.dumps(value, ensure_ascii=False)
        arguments = self._argument_prefix(self._parameter_name) + encoded_value
        self._reset_parameter()
        return [
            {
                "index": self._current["index"],
                "function": {"arguments": arguments},
            }
        ]

    def _finish_function(self) -> list[dict]:
        if self._current is None:
            return []

        arguments = "}" if self._current["has_arguments"] else "{}"
        delta = {
            "index": self._current["index"],
            "function": {"arguments": arguments},
        }
        self._current = None
        self.completed_calls += 1
        return [delta]


def parse_toolcalls(text: str) -> list[ToolCall]:
    # If there are outer <tool_call> wrappers, unwrap them; otherwise use the whole text
    segments: list[tuple[str, str]] = []  # (raw, inner)
    outer_matches = list(_OUTER.finditer(text))
    if outer_matches:
        is_wrapped = False
        for m in outer_matches:
            segments.append((m.group(0), m.group(1)))
    else:
        # No outer wrapper — look for bare <function=...> blocks
        is_wrapped = True
        segments = [(text, text)]

    results = []
    for _, inner in segments:
        for fm in _FUNC.finditer(inner):
            func_name = fm.group(1)
            func_body = fm.group(2)
            args: dict[str, any] = {}
            for pm in _PARAM.finditer(func_body):
                key = pm.group(1).strip()
                val = pm.group(2).strip()
                val = coerce_param_value(val)
                args[key] = val

            args_json = json.dumps(args, ensure_ascii=False)
            results.append(ToolCall(function=Tool(name=func_name, arguments=args_json)))

    xlogger.debug(
        f"qwen3_coder: Parsed {len(results)} tool calls",
        {"raw_text": text, "results": results, "is_wrapped": is_wrapped},
    )
    return results
