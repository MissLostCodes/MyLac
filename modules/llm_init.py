import os
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from modules.logger import get_logger
logger = get_logger("AGENT")


# --- LLM Loader ---
async def create_llm(
    model_name: str,
    api_key: str = None,
    model_provider: str = "ollama",
):

    model_temperature = float(os.getenv("MODEL_TEMPERATURE", 0.7))

    if model_provider == "ollama":
        logger.info("🤖 Loading Ollama model...")
        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL"), temperature=model_temperature
        )
    elif model_provider == "google":
        logger.info("🤖 Loading Google model...")
        return ChatGoogleGenerativeAI(
            model=model_name or os.getenv("GOOGLE_MODEL"),
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            temperature=model_temperature,
        )

    else:
        logger.error(f"Unknown model provider: {model_provider}")
        raise ValueError(f"Unknown model provider: {model_provider}")

