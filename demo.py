import logging
import asyncio
from pfun_llm.backend.ollama import OllamaGenerativeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


async def main():
    model = OllamaGenerativeModel()  # or pass model="<name>" explicitly
    response = await model.generate_content("Hello!")
    print(response)


asyncio.run(main())
