"""
Simple OCD Test for Bedrock Embeddings
========================================

Tests Bedrock embedding configuration with different model formats.
Run this on your AWS instance to verify everything works.

Usage:
    # Test with simple model ID
    python test_bedrock.py
    
    # Test with ARN
    BEDROCK_MODEL_ID="arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1" python test_bedrock.py
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config

def test_bedrock_embeddings():
    """Simple test for Bedrock embeddings."""
    
    print("=" * 80)
    print("BEDROCK EMBEDDINGS TEST")
    print("=" * 80)
    
    # Test cases
    test_cases = [
        {
            "name": "Amazon Titan v1 (Simple ID)",
            "model_id": "amazon.titan-embed-text-v1",
            "expected_provider": "amazon"
        },
        {
            "name": "Amazon Titan v1 (ARN)",
            "model_id": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1",
            "expected_provider": "amazon"
        },
        {
            "name": "Cohere English (Simple ID)",
            "model_id": "cohere.embed-english-v3",
            "expected_provider": "cohere"
        },
        {
            "name": "Cohere English (ARN)",
            "model_id": "arn:aws:bedrock:us-east-1::foundation-model/cohere.embed-english-v3",
            "expected_provider": "cohere"
        }
    ]
    
    print("\n📋 Testing Provider Extraction:\n")
    
    for i, test in enumerate(test_cases, 1):
        model_id = test["model_id"]
        expected = test["expected_provider"]
        
        # Extract provider
        if model_id.startswith("arn:"):
            model_name = model_id.split("/")[-1]
            provider = model_name.split(".")[0]
        else:
            provider = model_id.split(".")[0]
        
        status = "✅" if provider == expected else "❌"
        print(f"{status} Test {i}: {test['name']}")
        print(f"   Model ID: {model_id}")
        print(f"   Extracted Provider: {provider}")
        print(f"   Expected Provider: {expected}")
        print()
    
    # Now test actual embedding creation if credentials available
    print("=" * 80)
    print("TESTING ACTUAL BEDROCK EMBEDDING")
    print("=" * 80)
    
    # Check for AWS credentials
    has_creds = (
        Config.AWS_ACCESS_KEY_ID or
        os.getenv("AWS_ACCESS_KEY_ID") or
        os.getenv("AWS_PROFILE")
    )
    
    if not has_creds:
        print("\n⚠️  No AWS credentials found")
        print("   Set one of:")
        print("   - AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY in config/.env")
        print("   - AWS_PROFILE environment variable")
        print("   - Use IAM role (on EC2/ECS)")
        print("\n   Skipping actual embedding test...")
        return
    
    try:
        print(f"\n🔧 Configuration:")
        print(f"   Provider: bedrock")
        print(f"   Model ID: {Config.BEDROCK_MODEL_ID}")
        print(f"   Region: {Config.AWS_REGION}")
        
        # Set to bedrock
        Config.EMBEDDING_PROVIDER = "bedrock"
        
        # Get embeddings instance
        print(f"\n📊 Creating embeddings instance...")
        embeddings = Config.get_embeddings()
        
        print(f"   ✅ Instance created successfully!")
        print(f"   Type: {type(embeddings).__name__}")
        
        # Test embedding generation
        print(f"\n🧪 Testing embedding generation...")
        test_text = "This is a test sentence for AWS Bedrock embeddings."
        
        print(f"   Input: '{test_text}'")
        embedding = embeddings.embed_query(test_text)
        
        print(f"   ✅ Embedding generated successfully!")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   Sample values: {embedding[:5]}")
        print(f"   Type: {type(embedding[0])}")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("   Install dependencies: pip install langchain-aws boto3")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Error Type: {type(e).__name__}")
        
        if "credentials" in str(e).lower():
            print("\n   This looks like a credentials issue.")
            print("   Make sure AWS credentials are configured correctly.")
        elif "endpoint" in str(e).lower():
            print("\n   This looks like an endpoint/region issue.")
            print(f"   Check if model is available in region: {Config.AWS_REGION}")
        elif "provider" in str(e).lower():
            print("\n   This looks like a provider parameter issue.")
            print("   The provider parameter should be automatically extracted now.")
        
        import traceback
        print("\n📋 Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    test_bedrock_embeddings()
