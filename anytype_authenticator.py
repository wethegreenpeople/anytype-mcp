import logging
import os
import json
from typing import Dict, Any

from anytype_store import AnyTypeStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnytypeAuthenticator:
    def __init__(self, store: AnyTypeStore, config_file: str):
        self.store = store
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file, creating it if it doesn't exist."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    def _save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f)
        except IOError as e:
            logger.error(f"Failed to save configuration: {e}")

    async def get_challenge_id(self) -> str:
        """
        Get a challenge ID
        
        Returns:
            An string that is a challenge ID
        """
        
        try:
            if not self.config.get('challenge_id'):
                # Get challenge ID
                self.config['challenge_id'] = await self.store.get_challenge()
                self._save_config()
                logger.info(f"Obtained challenge ID")

                return self.config['challenge_id']
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    async def get_token(self, secret_code: str) -> AnyTypeStore:
        """
        Get a valid anytype authentication token using a challenge id and secret code
        
        Returns:
            A valid anytype app token
        """

        try:
            # Get app token
            app_token = await self.store.get_token(self.config['challenge_id'], secret_code)
            
            # Save the token
            self.config['app_token'] = app_token
            self._save_config()

            return app_token

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    def get_authenticated_store(self) -> AnyTypeStore:
        """
        Get an authenticated Anytype store, performing authentication if needed.
        
        Returns:
            An AnyTypeStore with a valid app token
        """
        # Check if we have a valid existing token
        if self.config.get('app_token'):
            self.store.app_token = self.config['app_token']
            return self.store

        raise Exception("No valid anytype app token found")
