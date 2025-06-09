import requests
import json

def test_patch_functions():
    url = "http://localhost:13337/mcp"
    headers = {"Content-Type": "application/json"}
    
    # Test patch_bytes
    print("Testing patch_bytes...")
    request = {
        "jsonrpc": "2.0",
        "method": "patch_bytes",
        "params": ["0x102F2FB9", "909090"],
        "id": 1
    }
    
    try:
        response = requests.post(url, headers=headers, json=request)
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test patch_instruction
    print("\nTesting patch_instruction...")
    request = {
        "jsonrpc": "2.0",
        "method": "patch_instruction", 
        "params": ["0x102F2FBC", "nop"],
        "id": 2
    }
    
    try:
        response = requests.post(url, headers=headers, json=request)
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test apply_patches
    print("\nTesting apply_patches...")
    request = {
        "jsonrpc": "2.0",
        "method": "apply_patches",
        "params": [],
        "id": 3
    }
    
    try:
        response = requests.post(url, headers=headers, json=request)
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_patch_functions()