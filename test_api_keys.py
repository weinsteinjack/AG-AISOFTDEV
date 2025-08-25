#!/usr/bin/env python3
"""
API Key Testing Script for AI-Driven Software Engineering Course
================================================================

This script tests all API keys configured in utils.py to verify they are working correctly.
It performs simple completion requests to each provider to validate connectivity and authentication.

Usage:
    python test_api_keys.py

Requirements:
    - All API keys should be configured in your .env file
    - Required packages should be installed (see utils.py for the list)
"""

import os
import sys
from datetime import datetime

# Import our utils functions
try:
    from utils import load_environment, RECOMMENDED_MODELS
except ImportError as e:
    print(f"❌ Error importing utils: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

def test_openai_api(api_key):
    """Test OpenAI API key with a simple completion request."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use cheaper model for testing
            messages=[{"role": "user", "content": "Say 'Hello' if you can read this."}],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content
        return True, result.strip()
        
    except ImportError:
        return False, "OpenAI library not installed. Run: pip install openai"
    except Exception as e:
        return False, str(e)

def test_anthropic_api(api_key):
    """Test Anthropic API key with a simple completion request."""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Use cheaper model for testing
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'Hello' if you can read this."}]
        )
        
        result = response.content[0].text
        return True, result.strip()
        
    except ImportError:
        return False, "Anthropic library not installed. Run: pip install anthropic"
    except Exception as e:
        return False, str(e)

def test_huggingface_api(api_key):
    """Test Hugging Face API key with a simple completion request."""
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=api_key)
        
        # Test with a small, fast model
        model_name = "microsoft/DialoGPT-medium"
        response = client.chat_completion(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        return True, result.strip()
        
    except ImportError:
        return False, "Hugging Face Hub library not installed. Run: pip install huggingface_hub"
    except Exception as e:
        return False, str(e)

def test_google_api(api_key):
    """Test Google API key with a simple completion request."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'Hello' if you can read this.")
        
        result = response.text
        return True, result.strip()
        
    except ImportError:
        return False, "Google Generative AI library not installed. Run: pip install google-generativeai"
    except Exception as e:
        return False, str(e)

def get_provider_models(provider):
    """Get all models for a specific provider from RECOMMENDED_MODELS."""
    return [model for model, config in RECOMMENDED_MODELS.items() if config['provider'] == provider]

def main():
    """Main testing function."""
    print("=" * 60)
    print("API Key Testing Script")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load environment variables
    load_environment()
    
    # Define test cases
    test_cases = [
        {
            'name': 'OpenAI',
            'env_var': 'OPENAI_API_KEY',
            'test_function': test_openai_api,
            'provider': 'openai'
        },
        {
            'name': 'Anthropic',
            'env_var': 'ANTHROPIC_API_KEY',
            'test_function': test_anthropic_api,
            'provider': 'anthropic'
        },
        {
            'name': 'Hugging Face',
            'env_var': 'HUGGINGFACE_API_KEY',
            'test_function': test_huggingface_api,
            'provider': 'huggingface'
        },
        {
            'name': 'Google (Gemini)',
            'env_var': 'GOOGLE_API_KEY',
            'test_function': test_google_api,
            'provider': 'gemini'
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        provider_name = test_case['name']
        env_var = test_case['env_var']
        test_func = test_case['test_function']
        provider = test_case['provider']
        
        print(f"Testing {provider_name}...")
        print("-" * 40)
        
        # Check if API key exists
        api_key = os.getenv(env_var)
        if not api_key:
            print(f"❌ {env_var} not found in environment variables")
            print(f"   Please add {env_var}=your_api_key to your .env file")
            results[provider_name] = False
        else:
            # Mask the API key for security
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            print(f"🔑 Found API key: {masked_key}")
            
            # Test the API
            success, message = test_func(api_key)
            if success:
                print(f"✅ {provider_name} API test successful!")
                print(f"   Response: {message}")
                results[provider_name] = True
                
                # Show available models for this provider
                models = get_provider_models(provider)
                if models:
                    print(f"   Available models ({len(models)}): {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
            else:
                print(f"❌ {provider_name} API test failed!")
                print(f"   Error: {message}")
                results[provider_name] = False
        
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    working_providers = [name for name, working in results.items() if working]
    failed_providers = [name for name, working in results.items() if not working]
    
    print(f"✅ Working providers ({len(working_providers)}): {', '.join(working_providers) if working_providers else 'None'}")
    print(f"❌ Failed providers ({len(failed_providers)}): {', '.join(failed_providers) if failed_providers else 'None'}")
    
    if working_providers:
        print(f"\n🎉 You can use models from: {', '.join(working_providers)}")
    
    if failed_providers:
        print(f"\n🔧 To fix failed providers:")
        for provider in failed_providers:
            test_case = next(tc for tc in test_cases if tc['name'] == provider)
            print(f"   - Check your {test_case['env_var']} in the .env file")
            print(f"   - Verify the API key is valid and has proper permissions")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return exit code based on results
    if any(results.values()):
        print("\n✅ At least one provider is working!")
        return 0
    else:
        print("\n❌ No providers are working. Please check your API keys.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
