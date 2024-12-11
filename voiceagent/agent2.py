import logging
import os
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, deepgram, silero
import asyncio

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

def prewarm(proc: JobProcess):
    """Prepares the environment before the job starts."""
    proc.userdata["vad"] = silero.VAD.load()
    # Store both user and agent transcripts in a dictionary
    proc.userdata["transcripts"] = {
        "user": [],
        "agent": []
    }

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
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
    )

    def on_user_transcript(transcript: str, **kwargs):
        """Handles user STT transcriptions."""
        logger.info(f"User said: {transcript}")
        ctx.proc.userdata["transcripts"]["user"].append(transcript)
        logger.info(f"Current User Transcripts: {ctx.proc.userdata['transcripts']['user']}")

    # This callback might be available in the VoicePipelineAgent.
    # If not, you will need to integrate it at a point where the agent produces the response.
    def on_agent_transcript(transcript: str, **kwargs):
        """Handles agent responses."""
        logger.info(f"Agent said: {transcript}")
        ctx.proc.userdata["transcripts"]["agent"].append(transcript)
        logger.info(f"Current Agent Transcripts: {ctx.proc.userdata['transcripts']['agent']}")

    # Attach the handlers to the assistant
    assistant.on_user_transcript = on_user_transcript
    assistant.on_agent_transcript = on_agent_transcript  # Only if VoicePipelineAgent supports this

    assistant.start(ctx.room, participant)

    async def on_participant_disconnected():
        """Handles participant disconnection and saves STT transcripts."""
        logger.info(f"Participant {participant.identity} has disconnected.")
        save_transcripts_to_file(ctx.proc.userdata["transcripts"])

    participant.on_disconnected = on_participant_disconnected  # Attach the disconnect event

    try:
        logger.info("Running assistant. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(10)  # Keeps the assistant running to detect disconnections
    except KeyboardInterrupt:
        logger.info("User pressed Ctrl+C. Shutting down.")
    finally:
        logger.info("Reached finally block, attempting to save transcripts...")
        save_transcripts_to_file(ctx.proc.userdata["transcripts"])


def save_transcripts_to_file(transcripts):
    """Saves the captured STT transcriptions from both user and agent to a file."""
    output_dir = "/Users/7one/Documents/Work/mangoesai/voiceagent"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "stt_transcripts.txt")

    logger.info(f"Attempting to save {len(transcripts['user']) + len(transcripts['agent'])} transcriptions to {output_file}")
    logger.info(f"User Transcripts Content: {transcripts['user']}")
    logger.info(f"Agent Transcripts Content: {transcripts['agent']}")

    try:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write("=== User Transcripts ===\n")
            for i, line in enumerate(transcripts["user"], start=1):
                file.write(f"User {i}: {line}\n")

            file.write("\n=== Agent Transcripts ===\n")
            for i, line in enumerate(transcripts["agent"], start=1):
                file.write(f"Agent {i}: {line}\n")

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
