"""
Standalone Bedrock Embeddings Test
===================================

Tests AWS Bedrock embeddings without dependencies on src module.
Designed for EC2 instances with IAM role permissions.

Usage:
    python test_bedrock_standalone.py
"""


# ============================================================================
# Configuration (Edit these values)
# ============================================================================

# Option 1: Simple model ID with explicit provider
# MODEL_ID = "us.amazon.titan-embed-text-v2:0"
# PROVIDER = "amazon"  # REQUIRED: amazon, cohere, anthropic, ai21, meta, mistral

# Option 2: ARN with explicit provider
MODEL_ID = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/xxxxxxxxxx"
PROVIDER = "amazon"

AWS_REGION = "us-east-1"

# ============================================================================
# Test Code
# ============================================================================


def test_bedrock_embeddings():
    """Test Bedrock embeddings using EC2 IAM role."""

    print("=" * 80)
    print("BEDROCK EMBEDDINGS STANDALONE TEST (EC2 IAM Role)")
    print("=" * 80)

    print("\n🔧 Configuration:")
    print(f"   Model ID: {MODEL_ID}")
    print(f"   Provider: {PROVIDER}")
    print(f"   Region: {AWS_REGION}")
    print("   Auth Method: EC2 IAM Role (automatic)")

    # Check dependencies
    print("\n📦 Checking dependencies...")
    try:
        import boto3
        from langchain_aws import BedrockEmbeddings

        print("   ✅ langchain-aws installed")
        print("   ✅ boto3 installed")
    except ImportError as e:
        print(f"   ❌ Missing dependency: {e}")
        print("\n   Install with:")
        print("   pip install langchain-aws boto3")
        return

    # Check IAM role credentials
    print("\n🔑 Checking IAM role credentials...")
    try:
        session = boto3.Session(region_name=AWS_REGION)
        credentials = session.get_credentials()

        if credentials:
            print("   ✅ IAM role credentials found")
            print(f"   Access Key ID: {credentials.access_key[:10]}...")

            # Get caller identity
            sts = session.client("sts")
            identity = sts.get_caller_identity()
            print(f"   Account: {identity['Account']}")
            print(f"   ARN: {identity['Arn']}")
        else:
            print("   ❌ No credentials found")
            print("\n   Make sure:")
            print("   - EC2 instance has an IAM role attached")
            print("   - IAM role has bedrock permissions")
            return

    except Exception as e:
        print(f"   ❌ Error checking credentials: {e}")
        print("\n   Make sure:")
        print("   - EC2 instance has an IAM role attached")
        print("   - Instance metadata service is accessible")
        return

    # Create Bedrock embeddings
    try:
        print("\n📊 Creating BedrockEmbeddings instance...")

        # Create boto3 client using IAM role (no explicit credentials needed)
        client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

        print("   ✅ Bedrock client created using IAM role")

        # Create embeddings instance
        embeddings = BedrockEmbeddings(model_id=MODEL_ID, provider=PROVIDER, client=client)

        print("   ✅ Embeddings instance created successfully!")
        print(f"   Type: {type(embeddings).__name__}")

        # Test embedding generation
        print("\n🧪 Testing embedding generation...")
        test_text = "This is a test sentence for AWS Bedrock embeddings."

        print(f"   Input: '{test_text}'")
        print("   Calling embed_query()...")

        embedding = embeddings.embed_query(test_text)

        print("\n   ✅ Embedding generated successfully!")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   Type: {type(embedding[0])}")
        print(f"   Sample values (first 5): {[f'{v:.6f}' for v in embedding[:5]]}")
        print(f"   Min value: {min(embedding):.6f}")
        print(f"   Max value: {max(embedding):.6f}")

        # Test batch embeddings
        print("\n🧪 Testing batch embeddings...")
        test_texts = [
            "First document about machine learning.",
            "Second document about databases.",
            "Third document about cloud computing.",
        ]

        batch_embeddings = embeddings.embed_documents(test_texts)

        print("   ✅ Batch embeddings generated!")
        print(f"   Number of documents: {len(batch_embeddings)}")
        print(f"   Dimensions per doc: {len(batch_embeddings[0])}")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\n💡 Your Bedrock configuration is working correctly!")
        print(f"   Model: {MODEL_ID}")
        print(f"   Provider: {PROVIDER}")
        print(f"   Region: {AWS_REGION}")
        print("   Auth: EC2 IAM Role")

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("   Install: pip install langchain-aws boto3")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Error Type: {type(e).__name__}")

        error_msg = str(e).lower()

        if (
            "credentials" in error_msg
            or "unauthorized" in error_msg
            or "access denied" in error_msg
        ):
            print("\n   💡 This looks like an IAM permissions issue.")
            print("\n   Required IAM permissions:")
            print("   {")
            print('     "Version": "2012-10-17",')
            print('     "Statement": [')
            print("       {")
            print('         "Effect": "Allow",')
            print('         "Action": [')
            print('           "bedrock:InvokeModel",')
            print('           "bedrock:InvokeModelWithResponseStream"')
            print("         ],")
            print('         "Resource": "*"')
            print("       }")
            print("     ]")
            print("   }")
            print("\n   Check:")
            print("   - EC2 instance has IAM role attached")
            print("   - IAM role has bedrock:InvokeModel permission")
            print(f"   - Model is available in region '{AWS_REGION}'")

        elif "model" in error_msg or "not found" in error_msg:
            print("\n   💡 This looks like a model availability issue.")
            print("   Check:")
            print(f"   - Model '{MODEL_ID}' exists in region '{AWS_REGION}'")
            print("   - You have access to this model")
            print("   - Model ID format is correct")

        elif "provider" in error_msg:
            print("\n   💡 This looks like a provider issue.")
            print("   Check:")
            print(f"   - PROVIDER is set correctly (currently: '{PROVIDER}')")
            print("   - Valid values: amazon, cohere, anthropic, ai21, meta, mistral")

        else:
            print("\n   💡 Unexpected error. Full traceback:")

        import traceback

        print("\n📋 Full traceback:")
        traceback.print_exc()

        print("\n" + "=" * 80)
        print("🔍 DEBUGGING TIPS FOR EC2")
        print("=" * 80)

        print("\n1. Check IAM role is attached:")
        print("   aws sts get-caller-identity")

        print("\n2. Verify model access:")
        print(f"   aws bedrock list-foundation-models --region {AWS_REGION}")

        print("\n3. Test bedrock-runtime access:")
        print(f"   aws bedrock-runtime list-foundation-models --region {AWS_REGION}")

        print("\n4. Check instance metadata service:")
        print("   curl http://169.254.169.254/latest/meta-data/iam/security-credentials/")

        print("\n5. Common model IDs:")
        print("   - amazon.titan-embed-text-v1")
        print("   - us.amazon.titan-embed-text-v2:0")
        print("   - cohere.embed-english-v3")
        print("   - cohere.embed-multilingual-v3")


if __name__ == "__main__":
    test_bedrock_embeddings()
