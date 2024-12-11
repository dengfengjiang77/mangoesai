import logging
import os
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, deepgram, silero
from playht import PlayHTTTS
import asyncio


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    """Prepares the environment before the job starts."""
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["stt_transcripts"] = []  # Unified list to store all STT transcripts


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the voice agent."""
    initial_ctx = llm.ChatContext().append(
        role="system",
        text="You are a voice assistant created by LiveKit. Your interface with users will be voice."
    )

    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant {participant.identity}")

    # Initialize the assistant with VAD, STT, LLM, and TTS capabilities
    assistant = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=PlayHTTTS(api_key="6a38bb3186e94032989766ecf115d3cc", user_id="qPEVXyXdEQP9kMMbthI9bZx5gGI2"), # change to play.it
        chat_ctx=initial_ctx,
    )

    def on_user_transcript(transcript: str, **kwargs):
        """Handles user STT transcriptions and appends them to a list."""
        logger.info(f"User said: {transcript}")
        ctx.proc.userdata["stt_transcripts"].append(transcript)  # Always store transcript in the list
        logger.info(f"Current Transcripts: {ctx.proc.userdata['stt_transcripts']}")

    # Attach the handler to the assistant
    assistant.on_user_transcript = on_user_transcript
    assistant.start(ctx.room, participant)

    async def on_participant_disconnected():
        """Handles participant disconnection and saves STT transcripts."""
        logger.info(f"Participant {participant.identity} has disconnected.")
        save_transcripts_to_file(ctx.proc.userdata["stt_transcripts"])

    participant.on_disconnected = on_participant_disconnected  # Attach the disconnect event

    try:
        logger.info("Running assistant. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(10)  # Keeps the assistant running to detect disconnections
    except KeyboardInterrupt:
        logger.info("User pressed Ctrl+C. Shutting down.")
    finally:
        logger.info("Saving transcripts before exiting...")
        save_transcripts_to_file(ctx.proc.userdata["stt_transcripts"])


def save_transcripts_to_file(transcripts):
    """Saves the captured STT transcriptions to a file."""
    output_dir = "/Users/7one/Documents/Work/mangoesai/voiceagent"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "stt_transcripts.txt")
    
    logger.info(f"Saving {len(transcripts)} transcriptions to {output_file}")
    logger.info(f"Transcripts Content: {transcripts}")  # Log contents of the transcript list

    try:
        with open(output_file, "w", encoding="utf-8") as file:
            for i, line in enumerate(transcripts, start=1):
                file.write(f"{i}: {line}\n")
        logger.info(f"Transcriptions saved successfully to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save transcriptions to {output_file}: {e}")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
