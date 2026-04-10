import asyncio

from config.config import config

async def main():
    await config.update_access_token()



if __name__ == "__main__":
    asyncio.run(main())