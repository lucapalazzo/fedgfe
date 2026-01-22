#!/usr/bin/env python3
"""
Test to check if flux2-klein can accept T5/CLIP embeddings as input via Ollama API.
"""

import requests
import json
import numpy as np

OLLAMA_BASE_URL = "http://localhost:11434"
FLUX_MODEL = "x/flux2-klein:latest"

def test_text_prompt():
    """Test standard text prompt (baseline)"""
    print("\n" + "="*80)
    print("TEST 1: Standard text prompt")
    print("="*80)

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": FLUX_MODEL,
        "prompt": "A beautiful sunset over mountains",
        "stream": False
    }

    print(f"Request: {json.dumps(payload, indent=2)}")
    response = requests.post(url, json=payload)

    print(f"\nStatus Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response keys: {result.keys()}")
        print(f"Success: {result.get('done', False)}")
        if 'response' in result:
            print(f"Response length: {len(result['response'])}")
    else:
        print(f"Error: {response.text}")

    return response

def test_embedding_input():
    """Test if we can pass pre-computed embeddings"""
    print("\n" + "="*80)
    print("TEST 2: Pre-computed embeddings as input")
    print("="*80)

    # Generate fake T5/CLIP embeddings (proper dimensions)
    fake_t5_embedding = np.random.randn(4096).tolist()  # T5-XXL dimension
    fake_clip_embedding = np.random.randn(768).tolist()  # CLIP dimension

    url = f"{OLLAMA_BASE_URL}/api/generate"

    # Try different payload formats
    test_cases = [
        {
            "name": "embeddings field",
            "payload": {
                "model": FLUX_MODEL,
                "embeddings": fake_t5_embedding,
                "stream": False
            }
        },
        {
            "name": "t5_embeddings + clip_embeddings fields",
            "payload": {
                "model": FLUX_MODEL,
                "t5_embeddings": fake_t5_embedding,
                "clip_embeddings": fake_clip_embedding,
                "stream": False
            }
        },
        {
            "name": "encoder_hidden_states field",
            "payload": {
                "model": FLUX_MODEL,
                "encoder_hidden_states": fake_t5_embedding,
                "stream": False
            }
        },
        {
            "name": "options with embeddings",
            "payload": {
                "model": FLUX_MODEL,
                "prompt": "",
                "options": {
                    "embeddings": fake_t5_embedding
                },
                "stream": False
            }
        }
    ]

    for test_case in test_cases:
        print(f"\n--- Test: {test_case['name']} ---")
        response = requests.post(url, json=test_case['payload'])
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('done', False)}")
            print(f"Response keys: {result.keys()}")
        else:
            print(f"Error: {response.text[:200]}")

def test_custom_parameters():
    """Test if we can pass custom generation parameters"""
    print("\n" + "="*80)
    print("TEST 3: Custom generation parameters")
    print("="*80)

    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": FLUX_MODEL,
        "prompt": "A beautiful sunset",
        "options": {
            "num_predict": 1,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42
        },
        "stream": False
    }

    print(f"Request: {json.dumps(payload, indent=2)}")
    response = requests.post(url, json=payload)

    print(f"\nStatus Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result.get('done', False)}")
        print(f"Response keys: {result.keys()}")
    else:
        print(f"Error: {response.text[:200]}")

def main():
    print("Testing Flux2-Klein model capabilities with Ollama API")
    print(f"Model: {FLUX_MODEL}")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")

    # Run tests
    test_text_prompt()
    test_embedding_input()
    test_custom_parameters()

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
Based on the Ollama API structure and model capabilities:

1. FLUX2-Klein via Ollama uses a simplified text-to-image API
2. The model template is {{ .Prompt }}, accepting only text prompts
3. Ollama's /api/generate endpoint doesn't expose low-level embedding inputs
4. The model internally uses Qwen3 text embedder (not CLIP+T5)

RECOMMENDATION:
To use custom T5/CLIP embeddings, you would need to:
- Access the model directly via Python (diffusers/transformers)
- OR use HuggingFace API directly
- OR modify Ollama to expose embedding inputs (requires forking)

For your federated learning use case with adapter outputs:
- Pass text prompts generated from adapter outputs
- OR use the model outside of Ollama for direct embedding control
    """)

if __name__ == "__main__":
    main()
