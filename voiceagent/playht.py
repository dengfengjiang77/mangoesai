import httpx
import io
import logging
import asyncio
import uuid
from livekit.agents.tts import SynthesizedAudio
from livekit.rtc import AudioFrame  # 导入RawAudioFrame

logger = logging.getLogger("playhtt-tts")

class EventEmitter:
    def __init__(self):
        self._events = {}

    def on(self, event_name: str):
        def decorator(callback):
            if event_name not in self._events:
                self._events[event_name] = []
            self._events[event_name].append(callback)
            return callback
        return decorator

    def emit(self, event_name: str, *args, **kwargs):
        if event_name in self._events:
            for callback in self._events[event_name]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"❌ Event '{event_name}' handling error: {e}")


class Capabilities:
    def __init__(self, streaming: bool):
        self.streaming = streaming


class PlayHTTTS(EventEmitter):
    def __init__(self, api_key: str, user_id: str, voice: str = "Emma"):
        super().__init__()
        self.api_url = "https://play.ht/api/v1"
        self.api_key = api_key
        self.user_id = user_id
        self.voice = voice
        self.voice_engine = "Play3.0-mini-http"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-User-ID": self.user_id,
            "Content-Type": "application/json",
        }
        self.capabilities = Capabilities(streaming=True)
        
        self._input_buffer = []
        self._is_input_done = False
        self.sample_rate = 44100
        self.num_channels = 1

    def stream(self):
        logger.info("🎉 Returning self as the stream object")
        return self

    def push_text(self, text: str):
        logger.info(f"📥 Pushing text input: {text}")
        self._input_buffer.append(text)

    async def flush(self):
        if self._input_buffer:
            paragraph = ' '.join(self._input_buffer)
            logger.info(f"📥 Flushing paragraph for TTS synthesis: {paragraph}")
            await self.synthesize_stream(paragraph)
            self._input_buffer = []

    def end_input(self):
        logger.info("🚪 Ending input, no further data will be received")
        self._is_input_done = True

    async def synthesize_stream(self, text: str):
        payload = {
            "voice": self.voice, 
            "content": text,
            "format": "mp3",
            "voice_engine": self.voice_engine
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{self.api_url}/convert", json=payload, headers=self.headers)
                if response.status_code != 200:
                    logger.error(f"❌ Request failed, status: {response.status_code}, response: {response.text}")
                    request_id = f"error-{uuid.uuid4().hex[:8]}"
                    raw_frame =AudioFrame(
                        samples_per_channel=80, 
                        sample_rate=self.sample_rate,
                        num_channels=self.num_channels,
                        data=b'\x00' * 160
                    )
                    yield SynthesizedAudio(frame=raw_frame, request_id=request_id)
                    return

                response_data = response.json()
                if "transcriptionId" not in response_data:
                    logger.error(f"❌ Could not get transcriptionId: {response_data}")
                    request_id = f"error-{uuid.uuid4().hex[:8]}"
                    raw_frame = AudioFrame(
                        samples_per_channel=80,
                        sample_rate=self.sample_rate,
                        num_channels=self.num_channels,
                        data=b'\x00' * 160
                    )
                    yield SynthesizedAudio(frame=raw_frame, request_id=request_id)
                    return

                transcription_id = response_data["transcriptionId"]
                async with client.stream("GET", f"{self.api_url}/stream/{transcription_id}", headers=self.headers) as stream:
                    async for chunk in stream.aiter_bytes():
                        samples_per_channel = len(chunk) // (2 * self.num_channels)
                        raw_frame = AudioFrame(
                            samples_per_channel=samples_per_channel,
                            sample_rate=self.sample_rate,
                            num_channels=self.num_channels,
                            data=chunk
                        )
                        yield SynthesizedAudio(frame=raw_frame, request_id=transcription_id)

            except Exception as e:
                logger.error(f"❌ API Request Error: {e}")
                request_id = f"error-{uuid.uuid4().hex[:8]}"
                raw_frame = AudioFrame(
                    samples_per_channel=80,
                    sample_rate=self.sample_rate,
                    num_channels=self.num_channels,
                    data=b'\x00' * 160
                )
                yield SynthesizedAudio(frame=raw_frame, request_id=request_id)

    async def __aiter__(self):
        logger.info("🔄 Starting __aiter__ to consume text buffer and stream TTS")
        while not self._is_input_done or self._input_buffer:
            if self._input_buffer:
                text = ' '.join(self._input_buffer)
                async for chunk in self.synthesize_stream(text):
                    yield chunk
                self._input_buffer = []

        logger.info("🔄 Closing stream...")

    async def aclose(self):
        logger.info("❌ Closing TTS stream and cleaning up resources")
        self._is_input_done = True
