"""
Configuration Module for RAG System
====================================

Manages environment-specific configurations for:
- Embedding models (HuggingFace, AWS Bedrock)
- Database connections (Local vs Remote)
- Test vs Production environments
"""

import os
from enum import Enum
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
config_dir = Path(__file__).parent.parent / "config"
env_file = config_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    HUGGINGFACE = "huggingface"
    BEDROCK = "bedrock"


class Environment(str, Enum):
    """Deployment environment."""
    LOCAL = "local"
    REMOTE = "remote"
    TEST = "test"


class Config:
    """Central configuration for RAG system."""
    
    # ============================================================================
    # Embedding Configuration
    # ============================================================================
    
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    
    # HuggingFace settings
    HUGGINGFACE_MODEL: str = os.getenv(
        "HUGGINGFACE_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    HUGGINGFACE_DEVICE: str = os.getenv("HUGGINGFACE_DEVICE", "cpu")
    
    # AWS Bedrock settings
    BEDROCK_MODEL_ID: str = os.getenv(
        "BEDROCK_MODEL_ID",
        "amazon.titan-embed-text-v1"
    )
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN: Optional[str] = os.getenv("AWS_SESSION_TOKEN")
    
    # ============================================================================
    # Database Configuration
    # ============================================================================
    
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "local")
    
    # Local database (for tests and local development)
    LOCAL_DB_HOST: str = os.getenv("LOCAL_DB_HOST", "localhost")
    LOCAL_DB_PORT: int = int(os.getenv("LOCAL_DB_PORT", "9002"))
    LOCAL_DB_NAME: str = os.getenv("LOCAL_DB_NAME", "postgres")
    LOCAL_DB_USER: str = os.getenv("LOCAL_DB_USER", "user")
    LOCAL_DB_PASSWORD: str = os.getenv("LOCAL_DB_PASSWORD", "root")
    
    # Remote database (for production)
    REMOTE_DB_HOST: str = os.getenv("REMOTE_DB_HOST", "")
    REMOTE_DB_PORT: int = int(os.getenv("REMOTE_DB_PORT", "5432"))
    REMOTE_DB_NAME: str = os.getenv("REMOTE_DB_NAME", "postgres")
    REMOTE_DB_USER: str = os.getenv("REMOTE_DB_USER", "user")
    REMOTE_DB_PASSWORD: str = os.getenv("REMOTE_DB_PASSWORD", "root")
    
    # ============================================================================
    # Index Configuration
    # ============================================================================
    
    DEFAULT_INDEX_TYPE: str = os.getenv("DEFAULT_INDEX_TYPE", "hnsw")
    DEFAULT_DISTANCE_METRIC: str = os.getenv("DEFAULT_DISTANCE_METRIC", "cosine")
    
    # ============================================================================
    # Performance Configuration
    # ============================================================================
    
    CONNECTION_POOL_MIN_SIZE: int = int(os.getenv("CONNECTION_POOL_MIN_SIZE", "2"))
    CONNECTION_POOL_MAX_SIZE: int = int(os.getenv("CONNECTION_POOL_MAX_SIZE", "10"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))
    
    # ============================================================================
    # Helper Methods
    # ============================================================================
    
    @classmethod
    def get_connection_string(cls, force_local: bool = False) -> str:
        """
        Get database connection string based on environment.
        
        Args:
            force_local: Force local connection (used in tests)
            
        Returns:
            PostgreSQL connection string
        """
        if force_local or cls.ENVIRONMENT == "test":
            # Always use local for tests
            host = cls.LOCAL_DB_HOST
            port = cls.LOCAL_DB_PORT
            name = cls.LOCAL_DB_NAME
            user = cls.LOCAL_DB_USER
            password = cls.LOCAL_DB_PASSWORD
        elif cls.ENVIRONMENT == "remote" and cls.REMOTE_DB_HOST:
            # Use remote for production
            host = cls.REMOTE_DB_HOST
            port = cls.REMOTE_DB_PORT
            name = cls.REMOTE_DB_NAME
            user = cls.REMOTE_DB_USER
            password = cls.REMOTE_DB_PASSWORD
        else:
            # Default to local
            host = cls.LOCAL_DB_HOST
            port = cls.LOCAL_DB_PORT
            name = cls.LOCAL_DB_NAME
            user = cls.LOCAL_DB_USER
            password = cls.LOCAL_DB_PASSWORD
        
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"
    
    @classmethod
    def get_embeddings(cls):
        """
        Get configured embedding model instance.
        
        Returns:
            Embeddings instance (HuggingFaceEmbeddings or BedrockEmbeddings)
        """
        provider = cls.EMBEDDING_PROVIDER.lower()
        
        if provider == "bedrock":
            try:
                from langchain_aws import BedrockEmbeddings
            except ImportError:
                raise ImportError(
                    "langchain-aws not installed. Install with: pip install langchain-aws boto3"
                )
            
            kwargs = {
                "model_id": cls.BEDROCK_MODEL_ID,
                "region_name": cls.AWS_REGION,
            }
            
            # Add credentials if provided
            if cls.AWS_ACCESS_KEY_ID:
                kwargs["credentials_profile_name"] = None  # Use explicit credentials
                import boto3
                session = boto3.Session(
                    aws_access_key_id=cls.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=cls.AWS_SECRET_ACCESS_KEY,
                    aws_session_token=cls.AWS_SESSION_TOKEN,
                    region_name=cls.AWS_REGION
                )
                kwargs["client"] = session.client("bedrock-runtime")
            
            return BedrockEmbeddings(**kwargs)
        
        else:  # Default to HuggingFace
            from langchain_huggingface import HuggingFaceEmbeddings
            
            return HuggingFaceEmbeddings(
                model_name=cls.HUGGINGFACE_MODEL,
                model_kwargs={"device": cls.HUGGINGFACE_DEVICE}
            )
    
    @classmethod
    def print_config(cls):
        """Print current configuration (excluding secrets)."""
        print("=" * 80)
        print("CURRENT CONFIGURATION")
        print("=" * 80)
        print(f"\n🔧 Environment: {cls.ENVIRONMENT}")
        print(f"📊 Embedding Provider: {cls.EMBEDDING_PROVIDER}")
        
        if cls.EMBEDDING_PROVIDER.lower() == "bedrock":
            print(f"   - Model ID: {cls.BEDROCK_MODEL_ID}")
            print(f"   - AWS Region: {cls.AWS_REGION}")
            print(f"   - Credentials: {'✓ Configured' if cls.AWS_ACCESS_KEY_ID else '✗ Using default'}")
        else:
            print(f"   - Model: {cls.HUGGINGFACE_MODEL}")
            print(f"   - Device: {cls.HUGGINGFACE_DEVICE}")
        
        print(f"\n💾 Database:")
        if cls.ENVIRONMENT == "test":
            print(f"   - Connection: LOCAL (forced for tests)")
        elif cls.ENVIRONMENT == "remote" and cls.REMOTE_DB_HOST:
            print(f"   - Connection: REMOTE")
            print(f"   - Host: {cls.REMOTE_DB_HOST}:{cls.REMOTE_DB_PORT}")
        else:
            print(f"   - Connection: LOCAL")
            print(f"   - Host: {cls.LOCAL_DB_HOST}:{cls.LOCAL_DB_PORT}")
        
        print(f"   - Database: {cls.LOCAL_DB_NAME if cls.ENVIRONMENT != 'remote' else cls.REMOTE_DB_NAME}")
        print(f"   - User: {cls.LOCAL_DB_USER if cls.ENVIRONMENT != 'remote' else cls.REMOTE_DB_USER}")
        
        print(f"\n⚡ Performance:")
        print(f"   - Pool Size: {cls.CONNECTION_POOL_MIN_SIZE}-{cls.CONNECTION_POOL_MAX_SIZE}")
        print(f"   - Batch Size: {cls.BATCH_SIZE}")
        print("=" * 80 + "\n")


# Convenience function for tests
def get_test_config():
    """Get configuration forced to local database (for tests)."""
    return {
        "connection_string": Config.get_connection_string(force_local=True),
        "embeddings": Config.get_embeddings()
    }
