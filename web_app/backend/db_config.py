"""
Database configuration for MySQL connection.
Store credentials securely using environment variables or config file.
"""
import os
from pathlib import Path

# Load from .env file if python-dotenv is available
# python-dotenv is required and should be installed via requirements.txt
try:
    from dotenv import load_dotenv
    # Try loading from web_app/backend/.env first (same directory as this file)
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"[DB] Loaded environment variables from {env_path}")
    else:
        # Also try project root .env as fallback
        root_env_path = Path(__file__).parent.parent.parent / '.env'
        if root_env_path.exists():
            load_dotenv(root_env_path, override=True)
            print(f"[DB] Loaded environment variables from {root_env_path}")
except ImportError:
    print("[DB] WARNING: python-dotenv not installed. Install it with: pip install python-dotenv")
    print("[DB] Environment variables must be set manually or via system environment.")

# Database configuration
# SECURITY: Database credentials MUST be provided via environment variables
# Never hardcode passwords in source code. Use .env file or environment variables.
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '148.113.31.151'),
    'port': int(os.getenv('DB_PORT', '3306')),
    'database': os.getenv('DB_NAME', 'appdb'),
    'user': os.getenv('DB_USER'),  # REQUIRED - no default
    'password': os.getenv('DB_PASSWORD'),  # REQUIRED - no default
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci',
    'autocommit': False,
    'connect_timeout': 10,
    'pool_size': 5,
    'pool_reset_session': True,
}

# Validate required credentials
if not DB_CONFIG['user'] or not DB_CONFIG['password']:
    raise ValueError(
        "Database credentials are required. Please set DB_USER and DB_PASSWORD "
        "environment variables or create a .env file in web_app/backend/"
    )

# Alternative host (for failover)
DB_SECONDARY_HOST = os.getenv('DB_SECONDARY_HOST', '148.113.31.151')
