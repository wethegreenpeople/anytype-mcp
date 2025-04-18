import asyncio
import os

import nltk
from platformdirs import user_data_dir
from anytype import ingest_documents
from utils.anytype_authenticator import AnytypeAuthenticator
from anytype_api.anytype_store import AnyTypeStore

async def main():
    # Now run ingestion
    persist_dir = user_data_dir("anytype-mcp", "John Singh")
    config_path = os.path.join(persist_dir, "anytype_config.json")

    auth = AnytypeAuthenticator(AnyTypeStore(None, None), config_path)
    if auth.config.get('app_token') == None:
        await auth.get_challenge_id_async()
        print("Secret Code: ")
        await auth.get_token_async(input())

    nltk_dir = os.path.join(persist_dir, "nltk-data")
    print(f"huh {nltk_dir}")
    nltk.download("punkt_tab", download_dir=nltk_dir)
    nltk.data.path.append(nltk_dir)
    
    result = await ingest_documents()
    print(f"Done ingesting docs! Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())