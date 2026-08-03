"""Startup script for EMA backend — forces SelectorEventLoop on Windows."""
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print(f"Policy set: {type(asyncio.get_event_loop_policy()).__name__}", flush=True)

# Import uvicorn AFTER setting policy.
from uvicorn import Config, Server


async def main():
    loop = asyncio.get_running_loop()
    print(f"Running loop: {type(loop).__name__}", flush=True)

    config = Config("backend.main:app", host="0.0.0.0", port=8000, loop="asyncio")
    server = Server(config=config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
