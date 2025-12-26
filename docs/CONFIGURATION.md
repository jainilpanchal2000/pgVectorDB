# Configuration Guide - AWS Bedrock & Remote Database

Complete setup guide for embedding providers and database connections.

---

## 🚀 Quick Start

### Default Setup (No Changes Needed)
Using HuggingFace + local database? **Everything works as before!**

### Switch to AWS Bedrock
```bash
# 1. Install
pip install langchain-aws boto3

# 2. Configure config/.env
EMBEDDING_PROVIDER=bedrock
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret

# 3. Run tests with bedrock
python test/test_suite.py --embedding bedrock

# 4. Run benchmarks
python eval/benchmark_all_methods.py
```

---

## ✅ Embedding Compatibility - No Breaking Changes!

Both `HuggingFaceEmbeddings` and `BedrockEmbeddings` implement the **same LangChain Embeddings interface**:

```python
# Both providers have identical methods:
class Embeddings:
    def embed_documents(texts: List[str]) -> List[List[float]]
    def embed_query(text: str) -> List[float]
    def aembed_documents(texts: List[str]) -> List[List[float]]  # async
    def aembed_query(text: str) -> List[float]  # async
```

✅ **Drop-in replacement** - No code changes needed!  
✅ **Same return types** - Both return `List[float]` for embeddings  
✅ **Async support** - Both support async operations  
✅ **Works with pgVectorDB** - Fully compatible

---

## 📁 Configuration File: `config/.env`

### HuggingFace (Default - Free & Local)
```dotenv
ENVIRONMENT=local
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_DEVICE=cpu

LOCAL_DB_HOST=localhost
LOCAL_DB_PORT=9002
LOCAL_DB_USER=user
LOCAL_DB_PASSWORD=root
```

### AWS Bedrock (Managed Service)
```dotenv
ENVIRONMENT=remote
EMBEDDING_PROVIDER=bedrock

# Option 1: Simple model ID
BEDROCK_MODEL_ID=amazon.titan-embed-text-v1

# Option 2: Full ARN (for cross-region/cross-account)
BEDROCK_MODEL_ID=arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1

AWS_REGION=us-east-1

# Option A: Explicit credentials
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

# Option B: Leave empty to use default AWS credentials
# (from ~/.aws/credentials, environment, or IAM role)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

REMOTE_DB_HOST=your-db.example.com
REMOTE_DB_PORT=5432
REMOTE_DB_USER=user
REMOTE_DB_PASSWORD=root
```

### Credential Priority (Bedrock)
1. **Explicit in .env** - `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
2. **AWS CLI** - `~/.aws/credentials`
3. **Environment variables** - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
4. **IAM Role** - On EC2/ECS/Lambda (recommended for production)

---

## 🧪 Testing with Different Embeddings

### Command-Line Arguments (NEW!)
```bash
# Default: uses config/.env
python test/test_suite.py

# Force HuggingFace
python test/test_suite.py --embedding huggingface

# Force Bedrock
python test/test_suite.py --embedding bedrock

# Custom Bedrock model
python test/test_suite.py --embedding bedrock --bedrock-model amazon.titan-embed-text-v2

# Test embedding functionality only
python test/test_suite.py --test-embeddings
```

### Why Command-Line Args?
- ✅ No hardcoding - flexible testing
- ✅ Override config/.env when needed
- ✅ CI/CD friendly
- ✅ Quick provider switching

---

## 🔍 Bedrock Models

### Using Model IDs (Simple)
```dotenv
# Provider auto-extracted from model name (e.g., "amazon", "cohere")
BEDROCK_MODEL_ID=amazon.titan-embed-text-v1
```

### Using ARNs (Cross-Region/Cross-Account)
```dotenv
# Full ARN format - provider auto-extracted from model name
BEDROCK_MODEL_ID=arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1

# With account ID
BEDROCK_MODEL_ID=arn:aws:bedrock:us-west-2:123456789012:foundation-model/cohere.embed-english-v3
```

**Note:** The `provider` parameter (e.g., "amazon", "cohere") is automatically extracted from the model name in both formats.

### Available Models

| Model ID | Provider | ARN Example | Dimensions | Cost/1K |
|----------|----------|-------------|------------|---------|
| `amazon.titan-embed-text-v1` | amazon | `arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1` | 1536 | $0.0001 |
| `amazon.titan-embed-text-v2` | amazon | `arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2` | 1024 | $0.0001 |
| `cohere.embed-english-v3` | cohere | `arn:aws:bedrock:us-east-1::foundation-model/cohere.embed-english-v3` | 1024 | $0.0001 |
| `cohere.embed-multilingual-v3` | cohere | `arn:aws:bedrock:us-east-1::foundation-model/cohere.embed-multilingual-v3` | 1024 | $0.0001 |

**Provider extraction:** The system automatically extracts the provider name (e.g., "amazon", "cohere") from the model ID or ARN.

📖 [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)

---

## 💾 Database Configuration

### Same credentials for both (as requested)
```dotenv
# Local database
LOCAL_DB_USER=user
LOCAL_DB_PASSWORD=root

# Remote database (same credentials)
REMOTE_DB_USER=user
REMOTE_DB_PASSWORD=root
```

### Tests Always Use Localhost ✅
```bash
# Even with ENVIRONMENT=remote in .env
python test/test_suite.py  # ← Always uses localhost!
```

**Reason:** `get_test_config()` forces local database for test isolation.

---

## 🔧 Verification Commands

### Check current config
```bash
python -c "from src.config import Config; Config.print_config()"
```

### Test database connection
```bash
python scripts/test_connection.py
```

### Test embeddings (skips Bedrock if no credentials)
```bash
python test/test_suite.py --test-embeddings
```

---

## 🆘 Troubleshooting

### "langchain-aws not installed"
```bash
pip install langchain-aws boto3
```

### "Unable to locate credentials" (Bedrock)
```bash
# Option 1: Set in config/.env
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

# Option 2: AWS CLI
aws configure

# Option 3: Use IAM role (on EC2/ECS)
# No configuration needed - automatically detected

# Option 4: Skip Bedrock test
python test/test_suite.py --embedding huggingface
```

### "Could not connect to endpoint" (Bedrock)
```bash
# Check region matches model availability
# Some models only available in specific regions

# Verify ARN format if using ARN
# Correct: arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1
# Wrong:   arn:aws:bedrock::us-east-1:foundation-model/amazon.titan-embed-text-v1
```

### ARN Format Issues
```bash
# Valid ARN formats:
# Format 1 (foundation models):
arn:aws:bedrock:REGION::foundation-model/MODEL_ID

# Format 2 (custom/fine-tuned):
arn:aws:bedrock:REGION:ACCOUNT_ID:foundation-model/MODEL_ID

# Example valid ARNs:
arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1
arn:aws:bedrock:us-west-2:123456789012:foundation-model/cohere.embed-english-v3

# Check your ARN:
python -c "from src.config import Config; print(Config.BEDROCK_MODEL_ID)"
```

### "Connection refused" (Database)
```bash
# Check Docker
docker ps

# Test connection
telnet localhost 9002

# Verify credentials in config/.env
```

---

## 📊 Environment Variables

| Variable | Options | Default | Description |
|----------|---------|---------|-------------|
| `ENVIRONMENT` | local, remote, test | local | Environment mode |
| `EMBEDDING_PROVIDER` | huggingface, bedrock | huggingface | Embedding service |
| `HUGGINGFACE_MODEL` | sentence-transformers/* | all-MiniLM-L6-v2 | HF model |
| `HUGGINGFACE_DEVICE` | cpu, cuda | cpu | Compute device |
| `BEDROCK_MODEL_ID` | See table above | amazon.titan-embed-text-v1 | Bedrock model |
| `AWS_REGION` | us-east-1, etc. | us-east-1 | AWS region |
| `LOCAL_DB_HOST` | hostname | localhost | Local DB |
| `LOCAL_DB_PORT` | port | 9002 | Local port |
| `REMOTE_DB_HOST` | hostname | - | Remote DB |
| `REMOTE_DB_PORT` | port | 5432 | Remote port |

---

## 📚 Code Examples

### Using Config Module
```python
from src.config import Config
from src.core import pgVectorDB, IndexType

# Auto-loads from config/.env
embeddings = Config.get_embeddings()
connection = Config.get_connection_string()

rag = pgVectorDB(
    collection_name="my_docs",
    embedding_model=embeddings,
    connection_string=connection,
    index_type=IndexType.HNSW
)
```
### Using ARN with Bedrock
```python
from src.config import Config

# Override with ARN
Config.BEDROCK_MODEL_ID = "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2"
Config.EMBEDDING_PROVIDER = "bedrock"

embeddings = Config.get_embeddings()
# Uses the ARN to access the model
```

### Cross-Region Bedrock Access
```python
# Access model in different region than your app
Config.BEDROCK_MODEL_ID = "arn:aws:bedrock:eu-west-1::foundation-model/cohere.embed-english-v3"
Config.AWS_REGION = "eu-west-1"

embeddings = Config.get_embeddings()
```
### Force Test Configuration
```python
from src.config import get_test_config

# Always returns local DB + configured embeddings
config = get_test_config()
embeddings = config["embeddings"]
connection = config["connection_string"]
```

### Override Provider Programmatically
```python
from src.config import Config

# Temporarily override
Config.EMBEDDING_PROVIDER = "bedrock"
embeddings = Config.get_embeddings()
```

---

## ✨ What Changed

### New Features
- ✅ AWS Bedrock embeddings
- ✅ Remote database support
- ✅ Command-line args for tests
- ✅ Environment-aware config
- ✅ Test isolation (always localhost)

### Files Added
- `src/config.py` - Configuration module

### Files Modified
- `requirements.txt` - Added langchain-aws, boto3
- `config/.env` - New options
- `test/test_suite.py` - CLI args
- `eval/benchmark_all_methods.py` - Uses config
- `scripts/test_connection.py` - Uses config
- `README.md` - Updated docs

### Removed Files
- `eval/example_embeddings.py` - Not needed
- `docs/CHANGES.md` - Consolidated here
- `SETUP.md` - Consolidated here

---

## 🎯 Common Use Cases

### Local Development
```dotenv
ENVIRONMENT=local
EMBEDDING_PROVIDER=huggingface
```
```bash
python test/test_suite.py
```

### Production (Bedrock + Remote DB)
```dotenv
ENVIRONMENT=remote
EMBEDDING_PROVIDER=bedrock
REMOTE_DB_HOST=prod-db.company.com
```
```bash
python eval/benchmark_all_methods.py
```

### CI/CD Testing
```bash
# Force local regardless of .env
python test/test_suite.py --embedding huggingface
```

---

## 🔒 Security

1. ✅ Never commit `config/.env` (in `.gitignore`)
2. ✅ Use IAM roles on AWS EC2 (not access keys)
3. ✅ Rotate credentials regularly
4. ✅ Use SSL for remote DB
5. ✅ Strong passwords

---

## 🔄 Migration from Old Code

### Before
```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
rag = pgVectorDB(
    embedding_model=embeddings,
    connection_string="postgresql+asyncpg://user:root@localhost:9002/postgres"
)
```

### After (Recommended)
```python
from src.config import Config

embeddings = Config.get_embeddings()
rag = pgVectorDB(
    embedding_model=embeddings,
    connection_string=Config.get_connection_string()
)
```

### Old Code Still Works! ✅
Backward compatible - no breaking changes.

---

## 📞 Support

- Run `Config.print_config()` to see current settings
- Check `python scripts/test_connection.py` for diagnostics
- Tests: `python test/test_suite.py --help`
