import logging
from pathlib import Path

# Set up logging before importing modules
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Anytype MCP server...")
    
    # Run the MCP server
    # mcp.run(transport='stdio')

if __name__ == "__main__":
    main()