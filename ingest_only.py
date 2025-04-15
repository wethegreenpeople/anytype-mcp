import asyncio
from anytype_old import ingest_documents

async def main():
    # Optional: Run your authentication first, e.g. 
    # await get_anytype_challenge_id()
    # user enters secret code somewhere
    # await get_anytype_token(secret_code)

    # Now run ingestion
    result = await ingest_documents()
    print(f"Done ingesting docs! Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())