"""LiteLLM custom provider that answers through an agent CLI.

Registered once per profile name (``claude_cli``, ``copilot_cli`` and any profile the
knowledge base declares), so every OpenKB path that goes through LiteLLM - the wiki
compiler, PageIndex, and the agents SDK used by query, chat, lint and skills - can run
on a CLI the operator is already signed in to, with no API key in OpenKB.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    Usage,
)

from openkb.llm_cli import governance, runner, transcript
from openkb.llm_cli.profiles import CliProfile


class CliBackend(CustomLLM):
    """One profile, one LiteLLM provider. Stateless apart from its configuration."""

    def __init__(
        self, profile: CliProfile, policy: governance.Policy, kb_dir: Path | None = None
    ) -> None:
        super().__init__()
        self.profile = profile
        self.policy = policy
        self.kb_dir = kb_dir

    # -- shared -----------------------------------------------------------------
    def _check(self) -> governance.Facts:
        facts = governance.observe(self.profile, self.policy)
        reasons = governance.evaluate(self.policy, facts)
        if reasons:
            raise CustomLLMError(
                status_code=403,
                message=f"{self.profile.name} refused by data_governance: " + " ".join(reasons),
            )
        return facts

    def _model(self, model: str) -> str:
        name = model.split("/", 1)[1] if model.startswith(self.profile.name + "/") else model
        return name or self.profile.default_model

    def _render(self, messages: list, optional_params: dict) -> transcript.Rendered:
        return transcript.render(
            messages,
            tools=optional_params.get("tools"),
            response_format=optional_params.get("response_format"),
            system_inline=self.profile.system == "inline",
        )

    def _response(
        self,
        model: str,
        rendered: transcript.Rendered,
        result: runner.CliResult,
        model_response: ModelResponse | None,
    ) -> ModelResponse:
        parsed = transcript.parse(result.text, rendered)
        tool_calls = [
            ChatCompletionMessageToolCall(
                id=c["id"],
                type="function",
                function=Function(name=c["function"]["name"], arguments=c["function"]["arguments"]),
            )
            for c in parsed.tool_calls
        ] or None
        prompt_tokens = (
            result.input_tokens
            if result.input_tokens is not None
            else transcript.estimate_tokens(rendered.system + rendered.prompt)
        )
        completion_tokens = (
            result.output_tokens
            if result.output_tokens is not None
            else transcript.estimate_tokens(result.text)
        )
        response = model_response or ModelResponse()
        response.choices = [
            Choices(
                index=0,
                finish_reason=parsed.finish_reason,
                message=Message(
                    role="assistant", content=parsed.content or None, tool_calls=tool_calls
                ),
            )
        ]
        response.model = f"{self.profile.name}/{model}"
        response.created = int(time.time())
        setattr(
            response,
            "usage",
            Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        return response

    def _chunks(self, response: ModelResponse) -> list[GenericStreamingChunk]:
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        usage_block = (
            {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            if usage
            else None
        )
        chunks: list[GenericStreamingChunk] = []
        content = choice.message.content or ""
        calls = choice.message.tool_calls or []
        if content:
            chunks.append(
                {
                    "text": content,
                    "tool_use": None,
                    "is_finished": not calls,
                    "finish_reason": "" if calls else choice.finish_reason,
                    "usage": None if calls else usage_block,
                    "index": 0,
                }
            )
        for position, call in enumerate(calls):
            last = position == len(calls) - 1
            chunks.append(
                {
                    "text": "",
                    "is_finished": last,
                    "finish_reason": choice.finish_reason if last else "",
                    "usage": usage_block if last else None,
                    "index": 0,
                    "tool_use": {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                        "index": position,
                    },
                }
            )
        if not chunks:
            chunks.append(
                {
                    "text": "",
                    "tool_use": None,
                    "is_finished": True,
                    "finish_reason": choice.finish_reason,
                    "usage": usage_block,
                    "index": 0,
                }
            )
        return chunks

    # -- LiteLLM entry points -----------------------------------------------------
    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Any,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        *args: Any,
        timeout: Any = None,
        **kwargs: Any,
    ) -> ModelResponse:
        facts = self._check()
        name = self._model(model)
        rendered = self._render(messages, optional_params)
        try:
            result = runner.run(
                self.profile, name, rendered.system, rendered.prompt, timeout=_seconds(timeout)
            )
        except (runner.CliBackendError, FileNotFoundError) as error:
            raise CustomLLMError(status_code=502, message=str(error)) from None
        governance.audit(
            self.kb_dir,
            self.policy,
            facts,
            name,
            rendered.system + rendered.prompt,
            result.text,
            result.duration,
            result.exit_code,
        )
        return self._response(name, rendered, result, model_response)

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Any,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        *args: Any,
        timeout: Any = None,
        **kwargs: Any,
    ) -> ModelResponse:
        facts = self._check()
        name = self._model(model)
        rendered = self._render(messages, optional_params)
        try:
            result = await runner.arun(
                self.profile, name, rendered.system, rendered.prompt, timeout=_seconds(timeout)
            )
        except (runner.CliBackendError, FileNotFoundError) as error:
            raise CustomLLMError(status_code=502, message=str(error)) from None
        governance.audit(
            self.kb_dir,
            self.policy,
            facts,
            name,
            rendered.system + rendered.prompt,
            result.text,
            result.duration,
            result.exit_code,
        )
        return self._response(name, rendered, result, model_response)

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Any,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[GenericStreamingChunk]:
        # A CLI answers in one piece; the answer is delivered as one stream so callers that
        # always stream (chat, run_streamed) work unchanged.
        response = self.completion(
            model,
            messages,
            api_base,
            custom_prompt_dict,
            ModelResponse(),
            print_verbose,
            encoding,
            api_key,
            logging_obj,
            optional_params,
            *args,
            **kwargs,
        )
        yield from self._chunks(response)

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Any,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[GenericStreamingChunk]:
        response = await self.acompletion(
            model,
            messages,
            api_base,
            custom_prompt_dict,
            ModelResponse(),
            print_verbose,
            encoding,
            api_key,
            logging_obj,
            optional_params,
            *args,
            **kwargs,
        )
        for chunk in self._chunks(response):
            yield chunk


def _seconds(timeout: Any) -> float | None:
    if isinstance(timeout, (int, float)) and timeout > 0:
        return float(timeout)
    read = getattr(timeout, "read", None)
    return float(read) if isinstance(read, (int, float)) and read > 0 else None
