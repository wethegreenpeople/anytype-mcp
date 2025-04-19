import httpx
import json
from typing import Dict, List, Optional, TypedDict, Union, cast
import os

class DocumentResponse(TypedDict, total=False):
    object: Dict[str, object]
    id: str
    name: str
    space_id: str

class QueryResult(TypedDict):
    data: List[Dict[str, object]]

class ChallengeResponse(TypedDict):
    challenge_id: str

class TokenResponse(TypedDict):
    app_key: str

class AnyTypeStore:
    def __init__(self, space_id: Optional[str], app_token: Optional[str]):
        self._http_client = httpx.AsyncClient()
        self._api_url = "http://localhost:31009/v1"
        self.app_token = os.environ.get('ANYTYPE_APP_TOKEN', None) if app_token == None else app_token
        self.space_id = os.environ.get('ANYTYPE_SPACE_ID', None) if space_id == None else space_id

    async def get_challenge_async(self) -> str:
        request_url = f"{self._api_url}/auth/display_code?app_name=AnyTypeMcpServers"
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json"
        }

        response = await self._http_client.post(request_url, headers=headers)
        response.raise_for_status()
        body = cast(ChallengeResponse, response.json())
        return body["challenge_id"]

    async def get_token_async(self, challenge_id: str, code: str) -> str:
        request_url = f"{self._api_url}/auth/token?challenge_id={challenge_id}&code={code}"
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json"
        }

        response = await self._http_client.post(request_url, headers=headers)
        response.raise_for_status()
        body = cast(TokenResponse, response.json())
        return body["app_key"]

    async def get_documents_async(self, offset: int, limit: int) -> Dict[str, List[Dict[str, object]]]:
        request_url = f"{self._api_url}/spaces/{self.space_id}/objects?offset={offset}&limit={limit}"
        headers = {
            "Accept": "*/*",
            "Authorization": f"Bearer {self.app_token}",
            "Content-Type": "application/json"
        }

        response = await self._http_client.get(request_url, headers=headers)
        response.raise_for_status()
        return response.json()

    async def query_documents_async(self, offset: int, limit: int, query: str) -> QueryResult:
        request_url = f"{self._api_url}/search?offset={offset}&limit={limit}"
        headers = {
            "Accept": "*/*",
            "Authorization": f"Bearer {self.app_token}",
            "Content-Type": "application/json"
        }
        payload = {"query": query}

        response = await self._http_client.post(request_url, headers=headers, content=json.dumps(payload))
        response.raise_for_status()
        return cast(QueryResult, response.json())

    async def get_document_async(self, document_id: str, space_id: Optional[str]) -> DocumentResponse:
        request_url = f"{self._api_url}/spaces/{self.space_id if space_id is None else space_id}/objects/{document_id}"
        headers = {
            "Accept": "*/*",
            "Authorization": f"Bearer {self.app_token}",
            "Content-Type": "application/json"
        }

        response = await self._http_client.get(request_url, headers=headers)
        response.raise_for_status()
        return cast(DocumentResponse, response.json())
