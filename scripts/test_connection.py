"""
Database Connection & Requirements Test Script
==============================================

Tests database connectivity and verifies all requirements are met.

Usage:
    python scripts/test_connection.py
    
    # Or with custom connection string
    python scripts/test_connection.py --conn "postgresql+asyncpg://user:pass@host:port/db"
"""

import sys
import os
import asyncio
import argparse
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Handle nested event loops (same as notebook)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # nest_asyncio not critical for standalone scripts

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{BOLD}{BLUE}{'=' * 80}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(80)}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 80}{RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_info(text: str):
    """Print info message."""
    print(f"{BLUE}ℹ {text}{RESET}")


async def test_database_connection(connection_string: str) -> Dict[str, Any]:
    """Test database connection using pgVectorDB (same as notebook)."""
    results = {
        'connected': False,
        'version': None,
        'extensions': {},
        'error': None
    }
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from src.core import pgVectorDB, IndexType
        
        # Create minimal embedding model (won't be used, just for initialization)
        print_info("Initializing test embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Use pgVectorDB to test connection (same as notebook)
        print_info("Testing database connection with pgVectorDB...")
        test_rag = pgVectorDB(
            collection_name="connection_test_temp",
            embedding_model=embeddings,
            connection_string=connection_string,
            index_type=IndexType.HNSW
        )
        
        # Initialize (this will test the connection)
        await test_rag.initialize(overwrite_existing=True)
        results['connected'] = True
        print_success("Database connection successful")
        
        # Get database info using raw connection
        import asyncpg
        conn_str = connection_string.replace('postgresql+asyncpg://', 'postgresql://')
        conn = await asyncpg.connect(conn_str)
        
        # Get PostgreSQL version
        version = await conn.fetchval('SELECT version();')
        results['version'] = version.split(',')[0]
        print_info(f"PostgreSQL version: {results['version']}")
        
        # Check extensions
        extensions_to_check = ['vector', 'pg_trgm', 'vectorscale']
        for ext in extensions_to_check:
            try:
                ext_version = await conn.fetchval(
                    'SELECT extversion FROM pg_extension WHERE extname = $1',
                    ext
                )
                if ext_version:
                    results['extensions'][ext] = ext_version
                    print_success(f"Extension '{ext}' installed (version {ext_version})")
                else:
                    results['extensions'][ext] = None
                    if ext == 'vectorscale':
                        print_warning(f"Extension '{ext}' not installed (optional for DiskANN)")
                    else:
                        print_error(f"Extension '{ext}' not installed")
            except Exception as e:
                results['extensions'][ext] = None
                if ext == 'vectorscale':
                    print_warning(f"Extension '{ext}' not available (optional)")
                else:
                    print_error(f"Extension '{ext}' check failed: {e}")
        
        await conn.close()
        
        # Clean up test table (use raw SQL since drop_collection may not exist)
        try:
            cleanup_conn = await asyncpg.connect(conn_str)
            await cleanup_conn.execute('DROP TABLE IF EXISTS connection_test_temp CASCADE')
            await cleanup_conn.close()
        except Exception:
            pass  # Ignore cleanup errors
        
        await test_rag.close()
        
    except ImportError as e:
        results['error'] = f"Missing package: {e}"
        print_error(f"Required package not installed: {e}")
    except Exception as e:
        results['error'] = str(e)
        print_error(f"Connection failed: {e}")
    
    return results


def test_python_packages() -> Dict[str, Any]:
    """Test if required Python packages are installed."""
    results = {
        'all_installed': True,
        'packages': {}
    }
    
    required_packages = {
        'langchain': 'langchain',
        'langchain_core': 'langchain-core',
        'langchain_postgres': 'langchain-postgres',
        'langchain_community': 'langchain-community',
        'langchain_huggingface': 'langchain-huggingface',
        'sqlalchemy': 'sqlalchemy',
        'asyncpg': 'asyncpg',
        'psycopg2': 'psycopg2-binary',
        'nest_asyncio': 'nest-asyncio',
        'numpy': 'numpy',
        'sentence_transformers': 'sentence-transformers'
    }
    
    print_header("Checking Python Packages")
    
    for package, pip_name in required_packages.items():
        try:
            if package == 'psycopg2':
                import psycopg2
            else:
                __import__(package)
            results['packages'][pip_name] = True
            print_success(f"{pip_name} installed")
        except ImportError:
            results['packages'][pip_name] = False
            results['all_installed'] = False
            print_error(f"{pip_name} NOT installed")
    
    return results


def test_environment_config() -> Dict[str, Any]:
    """Test environment configuration."""
    results = {
        'config_found': False,
        'variables': {}
    }
    
    print_header("Checking Configuration")
    
    # Check for config/.env file
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
    
    if os.path.exists(config_path):
        results['config_found'] = True
        print_success(f"Configuration file found: config/.env")
        
        # Read .env file
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        if key in ['DB_CONNECTION_STRING', 'DB_HOST', 'DB_PORT', 'DB_NAME', 
                                   'DB_USER', 'EMBEDDING_MODEL']:
                            # Mask sensitive info
                            if 'PASSWORD' in key or 'CONNECTION_STRING' in key:
                                results['variables'][key] = '***' if value.strip() else 'NOT SET'
                            else:
                                results['variables'][key] = value.strip() if value.strip() else 'NOT SET'
                            
                            if value.strip():
                                print_success(f"{key} configured")
                            else:
                                print_warning(f"{key} is empty")
        except Exception as e:
            print_error(f"Error reading config file: {e}")
    else:
        results['config_found'] = False
        print_error("Configuration file NOT found: config/.env")
        print_info("Copy config/.env.example to config/.env and update with your settings")
    
    return results


def test_embedding_model() -> Dict[str, Any]:
    """Test embedding model."""
    results = {
        'model_available': False,
        'model_name': None,
        'dimensions': None
    }
    
    print_header("Checking Embedding Model")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print_info(f"Loading embedding model: {model_name}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Test embedding
        test_text = "This is a test sentence."
        embedding = embeddings.embed_query(test_text)
        
        results['model_available'] = True
        results['model_name'] = model_name
        results['dimensions'] = len(embedding)
        
        print_success(f"Embedding model loaded successfully")
        print_info(f"Vector dimensions: {results['dimensions']}")
        
    except Exception as e:
        results['error'] = str(e)
        print_error(f"Embedding model failed to load: {e}")
    
    return results


async def test_pgvectordb_import() -> Dict[str, Any]:
    """Test pgVectorDB import."""
    results = {
        'import_success': False,
        'available_classes': []
    }
    
    print_header("Checking pgVectorDB")
    
    try:
        from src.core import pgVectorDB, IndexType, StorageLayout, DistanceMetric
        
        results['import_success'] = True
        results['available_classes'] = ['pgVectorDB', 'IndexType', 'StorageLayout', 'DistanceMetric']
        
        print_success("pgVectorDB imported successfully")
        print_info(f"Available classes: {', '.join(results['available_classes'])}")
        print_info(f"Index types: {', '.join([idx.value for idx in IndexType])}")
        
    except Exception as e:
        results['error'] = str(e)
        print_error(f"pgVectorDB import failed: {e}")
    
    return results


def print_summary(all_results: Dict[str, Any]):
    """Print summary of all tests."""
    print_header("Test Summary")
    
    total_tests = 0
    passed_tests = 0
    
    # Python packages
    total_tests += 1
    if all_results['packages']['all_installed']:
        passed_tests += 1
        print_success("Python packages: ALL INSTALLED")
    else:
        missing = [pkg for pkg, installed in all_results['packages']['packages'].items() if not installed]
        print_error(f"Python packages: {len(missing)} MISSING")
        print_info(f"Missing: {', '.join(missing)}")
    
    # Configuration
    total_tests += 1
    if all_results['config']['config_found']:
        passed_tests += 1
        print_success("Configuration: FOUND")
    else:
        print_error("Configuration: NOT FOUND")
    
    # Database connection
    total_tests += 1
    if all_results['database']['connected']:
        passed_tests += 1
        print_success("Database connection: SUCCESSFUL")
    else:
        print_error("Database connection: FAILED")
    
    # Required extensions
    total_tests += 1
    required_ext = ['vector', 'pg_trgm']
    ext_ok = all(all_results['database']['extensions'].get(ext) for ext in required_ext)
    if ext_ok:
        passed_tests += 1
        print_success("Required extensions: INSTALLED")
    else:
        print_error("Required extensions: MISSING")
    
    # Embedding model
    total_tests += 1
    if all_results['embedding']['model_available']:
        passed_tests += 1
        print_success("Embedding model: LOADED")
    else:
        print_error("Embedding model: FAILED")
    
    # pgVectorDB
    total_tests += 1
    if all_results['pgvectordb']['import_success']:
        passed_tests += 1
        print_success("pgVectorDB: IMPORTED")
    else:
        print_error("pgVectorDB: FAILED")
    
    # Overall status
    print(f"\n{BOLD}Overall: {passed_tests}/{total_tests} tests passed{RESET}")
    
    if passed_tests == total_tests:
        print(f"\n{GREEN}{BOLD}✓ ALL TESTS PASSED - System ready!{RESET}\n")
        return 0
    else:
        print(f"\n{RED}{BOLD}✗ SOME TESTS FAILED - Check errors above{RESET}\n")
        return 1


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test pgVectorDB system requirements')
    parser.add_argument(
        '--conn',
        type=str,
        help='Database connection string (default: from config/.env)'
    )
    args = parser.parse_args()
    
    print_header("pgVectorDB System Requirements Test")
    
    all_results = {}
    
    # Test Python packages
    all_results['packages'] = test_python_packages()
    
    # Test configuration
    all_results['config'] = test_environment_config()
    
    # Get connection string
    connection_string = args.conn
    if not connection_string:
        # Try to read from config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith('DB_CONNECTION_STRING='):
                            connection_string = line.split('=', 1)[1].strip()
                            break
            except:
                pass
    
    if not connection_string:
        connection_string = "postgresql+asyncpg://user:root@localhost:9002/postgres"
        print_warning(f"Using default connection string: {connection_string}")
    
    # Test database connection
    print_header("Testing Database Connection")
    all_results['database'] = await test_database_connection(connection_string)
    
    # Test embedding model
    all_results['embedding'] = test_embedding_model()
    
    # Test pgVectorDB import
    all_results['pgvectordb'] = await test_pgvectordb_import()
    
    # Print summary
    exit_code = print_summary(all_results)
    
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
