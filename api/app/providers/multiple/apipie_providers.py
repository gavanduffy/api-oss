# api/app/providers/multiple/apipie_provider.py

import time
import httpx
import traceback
import ujson
from dataclasses import dataclass
from fastapi import Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Union, AsyncGenerator, Tuple, Optional, Iterable
from motor.motor_asyncio import AsyncIOMotorClient # Import for MongoDB connection
from ...responses import PrettyJSONResponse, StreamingResponseWithStatusCode
from ...core import UserManager, ProviderManager, settings
from ...utils import RequestProcessor
from ..ai_models import Model
from ..base_provider import BaseProvider, ProviderConfig
from ..utils import WebhookManager, ErrorHandler
from starlette.background import BackgroundTask # Added for StreamingResponseWithStatusCode

@dataclass(frozen=True)
class ApipieProviderConfig:
    """Configuration for Apipie API."""
    base_url: str = 'https://apipie.ai/v1'
    provider_id: str = 'apipie' # A unique ID for your provider, typically lowercase
    timeout: int = 100
    long_timeout: int = 10000

class SubProviderManager:
    """
    Manages sub-providers (API keys) for a given main provider.
    This class handles selecting an available API key from your database.
    You might need to adjust this based on how you manage your API keys.
    """
    def __init__(self, db_client: Any, provider_name: str):
        self.collection = db_client['db']['sub_providers']
        self.provider_name = provider_name

    async def get_available_provider(self, model: str) -> Optional[Dict[str, Any]]:
        # Implement logic to get an available API key/sub-provider from your database
        # This is a placeholder. You'll likely need to query based on model support and availability.
        sub_providers = await self.collection.find({
            'main_provider': self.provider_name,
            'models.api_name': {'$in': [model]},
            '$or': [
                {'working': True},
                {'working': {'$exists': False}}
            ]
        }).to_list(length=None)

        if not sub_providers:
            return None
        
        # Example: Simple round-robin or least used selection
        return min(
            sub_providers,
            key=lambda x: (x.get('usage', 0), x.get('last_used', 0))
        )

    async def update_provider(
        self,
        api_key: str,
        new_data: Dict[str, Any]
    ) -> None:
        # Update the sub-provider's metrics or status in the database
        update_data = {k: v for k, v in new_data.items() if k != '_id'}
        await self.collection.update_many(
            filter={'api_key': api_key},
            update={'$set': update_data}
        )

    async def disable_provider(
        self,
        api_key: str
    ) -> None:
        # Disable a sub-provider, e.g., if its API key is invalid or rate-limited
        await self.collection.update_many(
            filter={'api_key': api_key},
            update={'$set': {'working': False}}
        )

class APIClient:
    """Handles HTTP requests to the Apipie API."""
    def __init__(self, config: ApipieProviderConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=self.config.timeout)

    async def make_request(
        self,
        endpoint: str,
        method: str,
        sub_provider: Dict[str, Any],
        data: Dict[str, Any] = None,
        stream: bool = False,
        files: Dict[str, Any] = None,
        long_timeout: bool = False
    ) -> httpx.Response:
        # Headers for Apipie API authentication (Bearer token)
        headers = {
            'Authorization': f'Bearer {sub_provider["api_key"]}',
            'Content-Type': 'application/json'
        }
        # Adjust Accept header based on endpoint
        if endpoint == 'audio/speech':
            headers['Accept'] = 'audio/*'
        elif endpoint == 'images/generations':
            headers['Accept'] = 'application/json'

        url = f'{self.config.base_url}/{endpoint}'

        if long_timeout:
            self.client.timeout = self.config.long_timeout

        if data and not files:
            return await self.client.send(
                self.client.build_request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data
                ),
                stream=stream
            )
        elif files:
            headers.pop('Content-Type', None)
            return await self.client.send(
                self.client.build_request(
                    method=method,
                    url=url,
                    headers=headers,
                    files=files
                ),
                stream=stream
            )
        
        return await self.client.send(
            self.client.build_request(
                method=method,
                url=url,
                headers=headers
            ),
            stream=stream
        )


class ResponseHandler:
    """Handles parsing and formatting responses from the Apipie API."""
    def __init__(self, config: ApipieProviderConfig):
        self.config = config

    def create_error_response(
        self,
        message: str = 'Something went wrong. Try again later.',
        status_code: int = 500,
        error_type: str = 'invalid_response_error'
    ) -> PrettyJSONResponse:
        return PrettyJSONResponse(
            content={
                'error': {
                    'message': message,
                    'provider_id': self.config.provider_id,
                    'type': error_type,
                    'code': status_code
                }
            },
            status_code=status_code
        )

    def create_completion_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        # Add your provider's ID to the response
        return {
            'provider_id': self.config.provider_id,
            **response_data
        }

    def create_audio_response(self, content: bytes) -> Response:
        return Response(
            content=content,
            media_type='audio/mpeg', # Adjust as needed for Apipie
            headers={'content-disposition': 'attachment;filename=audio.mp3'}
        )

class MetricsManager:
    """Manages metrics and credit deduction for the Apipie provider."""
    def __init__(
        self,
        user_manager: UserManager,
        provider_manager: ProviderManager,
        sub_provider_manager: SubProviderManager
    ):
        self.user_manager = user_manager
        self.provider_manager = provider_manager
        self.sub_provider_manager = sub_provider_manager
        self.request_processor = RequestProcessor()

    async def update_user_credits(
        self,
        request: Request,
        model: str,
        token_count: int
    ) -> None:
        model_instance = Model.get_model(model)
        # Deduct credits based on the model's price and multiplier
        request.state.user['credits'] -= token_count * model_instance.pricing.multiplier
        await self.user_manager.update_user(request.state.user['user_id'], request.state.user)

    async def update_metrics(
        self,
        request: Request,
        model: str,
        sub_provider: Dict[str, Any],
        response_data: Dict[str, Any],
        start_time: float
    ) -> None:
        elapsed = time.time() - start_time
        # Implement logic to calculate word_count and token_count from response_data for Apipie
        word_count, token_count = self._calculate_counts(response_data)
        
        await self._update_provider_metrics(request, model, elapsed, word_count)
        await self._update_sub_provider_metrics(sub_provider)
        await self.update_user_credits(request, model, token_count)

    def _calculate_counts(self, response_data: Dict[str, Any]) -> Tuple[int, int]:
        # **IMPORTANT**: Implement how to count words/tokens based on Apipie's response structure
        word_count = 0
        token_count = 0
        if 'choices' in response_data and isinstance(response_data['choices'], list):
            for choice in response_data['choices']:
                if 'message' in choice and 'content' in choice['message']:
                    content = choice['message']['content']
                    if isinstance(content, str):
                        word_count += len(content.split())
                        token_count += self.request_processor.count_message_tokens(content)
        return word_count, token_count

    async def update_streaming_metrics(
        self,
        request: Request,
        model: str,
        sub_provider: Dict[str, Any],
        start_time: float
    ) -> None:
        elapsed = time.time() - start_time
        await self._update_provider_metrics(request, model, elapsed, 1)
        await self._update_sub_provider_metrics(sub_provider)

    async def _update_provider_metrics(
        self,
        request: Request,
        model: str,
        elapsed: float,
        word_count: int
    ) -> None:
        latency = (elapsed / word_count) if word_count > 0 else 0

        request.state.provider['usage'][model] = request.state.provider['usage'].get(model, 0) + 1
        request.state.provider['latency'][model] = (
            (request.state.provider['latency'].get(model, 0) + latency) / 2
            if request.state.provider['latency'].get(model, 0) != 0
            else latency
        )

        await self.provider_manager.update_provider(
            request.state.provider_name,
            request.state.provider,
            model
        )

    async def _update_sub_provider_metrics(
        self,
        sub_provider: Dict[str, Any]
    ) -> None:
        sub_provider['usage'] = sub_provider.get('usage', 0) + 1
        sub_provider['last_used'] = time.time()
        await self.sub_provider_manager.update_provider(
            sub_provider['api_key'],
            sub_provider
        )

class StreamHandler:
    """Handles streaming responses from the Apipie API."""
    def __init__(
        self,
        metrics_manager: MetricsManager,
        config: ApipieProviderConfig
    ):
        self.metrics_manager = metrics_manager
        self.config = config
        self.request_processor = RequestProcessor()

    async def handle_stream(
        self,
        response: httpx.Response,
        request: Request,
        model: str,
        sub_provider: Dict[str, Any],
        start_time: float
    ) -> StreamingResponse:
        async def stream_generator() -> AsyncGenerator[str, None]:
            await self.metrics_manager.update_streaming_metrics(
                request, model, sub_provider, start_time
            )

            async for line in response.aiter_lines():
                # **IMPORTANT**: Process each line of the streaming response according to Apipie's format.
                # This example assumes 'data: ' prefix and '[DONE]' marker.
                if line.startswith('data: ') and not line.startswith('data: [DONE]'):
                    yield await self._process_chunk(request, model, line)
                elif line.strip():
                    try:
                        parsed_chunk = {
                            'provider_id': self.config.provider_id,
                            **ujson.loads(line.strip())
                        }
                        token_count = sum(
                            self.request_processor.count_tokens(choice.get('delta', {}).get('content', ''))
                            for choice in parsed_chunk.get('choices', [{}])
                        )
                        await self.metrics_manager.update_user_credits(request, model, token_count)
                        yield f'data: {ujson.dumps(parsed_chunk)}\n\n'
                    except ujson.JSONDecodeError:
                        yield f'{line}\n\n'

            yield 'data: [DONE]\n\n'

        return StreamingResponseWithStatusCode(
            content=stream_generator(),
            media_type='text/event-stream',
            background=BackgroundTask(response.aclose)
        )

    async def _process_chunk(
        self,
        request: Request,
        model: str,
        line: str
    ) -> str:
        try:
            parsed_chunk = {
                'provider_id': self.config.provider_id,
                **ujson.loads(line[6:].strip()) # Remove 'data: ' prefix
            }
            token_count = sum(
                self.request_processor.count_tokens(choice.get('delta', {}).get('content', ''))
                for choice in parsed_chunk.get('choices', [{}])
            )

            await self.metrics_manager.update_user_credits(request, model, token_count)

            return f'data: {ujson.dumps(parsed_chunk)}\n\n'
        except Exception as e:
            print(f"Error processing stream chunk: {e} - Line: {line}")
            error_chunk = ujson.dumps({
                'error': {
                    'message': f"Failed to parse stream chunk: {str(e)}",
                    'type': 'stream_parsing_error',
                    'code': 500
                }
            })
            return f'data: {error_chunk}\n\n'


class EndpointHandler:
    """
    Handles the specific API calls for different functionalities for Apipie.
    """
    def __init__(
        self,
        api_client: APIClient,
        response_handler: ResponseHandler,
        metrics_manager: MetricsManager,
        stream_handler: StreamHandler,
        sub_provider_manager: SubProviderManager,
        provider_manager: ProviderManager,
        api_config: ApipieProviderConfig,
        provider_config: ProviderConfig
    ):
        self.api_client = api_client
        self.response_handler = response_handler
        self.metrics_manager = metrics_manager
        self.stream_handler = stream_handler
        self.sub_provider_manager = sub_provider_manager
        self.provider_manager = provider_manager
        self.api_config = api_config
        self.provider_config = provider_config
    
    async def _handle_error(
        self,
        request: Request,
        model: str,
        text: str
    ) -> None:
        await WebhookManager.send_to_webhook(
            request=request,
            is_error=True,
            model=model,
            pid=self.api_config.provider_id,
            exception=f'Error: {text}'
        )
        
        current_failure_count = request.state.provider['failures'].get(model, 0)
        request.state.provider['failures'][model] = current_failure_count + 1
        
        await self.provider_manager.update_provider(
            self.provider_config.name,
            request.state.provider
        )
    
    async def _handle_api_error(
        self,
        response: httpx.Response,
        stream: bool,
        sub_provider: Dict[str, Any],
        request: Request,
        model: str
    ) -> PrettyJSONResponse:
        if response.status_code in [401, 403, 404, 429]:
            await self.sub_provider_manager.disable_provider(
                sub_provider['api_key']
            )

        await self._handle_error(
            request,
            model,
            (await response.aread()).decode() if stream else response.text
        )
        return self.response_handler.create_error_response(
            message=(await response.aread()).decode() if stream else response.text,
            status_code=response.status_code,
            error_type='api_error'
        )

    async def handle_chat_completion(
        self,
        request: Request,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool,
        sub_provider: Dict[str, Any],
        start_time: float,
        **kwargs
    ) -> Union[PrettyJSONResponse, StreamingResponse]:
        # **IMPORTANT**: Adjust data payload as per Apipie's chat completion API documentation
        payload = {
            'model': model,
            'messages': messages,
            'stream': stream,
            **kwargs
        }

        response = await self.api_client.make_request(
            endpoint='chat/completions', # Adjust endpoint if different for Apipie
            method='POST',
            sub_provider=sub_provider,
            data=payload,
            stream=stream
        )

        if response.status_code != 200:
            return await self._handle_api_error(
                response, stream, sub_provider, request, model
            )

        if stream:
            return await self.stream_handler.handle_stream(
                response, request, model, sub_provider, start_time
            )
        
        json_response = response.json()
        await self.metrics_manager.update_metrics(
            request, model, sub_provider, json_response, start_time
        )
        return PrettyJSONResponse(
            self.response_handler.create_completion_response(json_response)
        )

    async def handle_images_generations(
        self,
        request: Request,
        model: str,
        prompt: str,
        sub_provider: Dict[str, Any],
        **kwargs
    ) -> PrettyJSONResponse:
        # Apipie images/generations payload from documentation
        payload = {
            "prompt": prompt,
            "model": model,
            "n": kwargs.get("n", 1),
            "size": kwargs.get("size", "1024x1024"),
            "quality": kwargs.get("quality", "standard"),
            "response_format": kwargs.get("response_format", "url"),
            "style": kwargs.get("style", None),
            "image": kwargs.get("image", None) # For image modification/update
        }
        # Remove None values from payload to avoid sending them if not provided
        payload = {k: v for k, v in payload.items() if v is not None}

        response = await self.api_client.make_request(
            endpoint='images/generations', # Adjust endpoint if different for Apipie
            method='POST',
            sub_provider=sub_provider,
            data=payload,
            long_timeout=True
        )

        if response.status_code != 200:
            return await self._handle_api_error(
                response, False, sub_provider, request, model
            )

        model_instance = Model.get_model(model)
        request.state.user['credits'] -= model_instance.pricing.price

        await self.metrics_manager.user_manager.update_user(
            request.state.user['user_id'],
            request.state.user
        )

        return PrettyJSONResponse(
            self.response_handler.create_completion_response(response.json())
        )
    
    async def handle_embeddings(
        self,
        request: Request,
        model: str,
        input_data: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]],
        sub_provider: Dict[str, Any],
        **kwargs
    ) -> PrettyJSONResponse:
        payload = {
            'model': model,
            'input': input_data,
            **kwargs
        }
        response = await self.api_client.make_request(
            endpoint='embeddings', # Adjust endpoint if different for Apipie
            method='POST',
            sub_provider=sub_provider,
            data=payload,
            long_timeout=True
        )

        if response.status_code != 200:
            return await self._handle_api_error(
                response, False, sub_provider, request, model
            )

        model_instance = Model.get_model(model)
        request.state.user['credits'] -= model_instance.pricing.price

        await self.metrics_manager.user_manager.update_user(
            request.state.user['user_id'],
            request.state.user
        )

        return PrettyJSONResponse(
            self.response_handler.create_completion_response(response.json())
        )
    
    async def handle_moderations(
        self,
        request: Request,
        model: str,
        input_data: Union[str, List[str]],
        sub_provider: Dict[str, Any]
    ) -> PrettyJSONResponse:
