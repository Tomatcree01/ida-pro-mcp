import requests
import json

def test_save_function():
    url = "http://localhost:13337/mcp"
    headers = {"Content-Type": "application/json"}
    
    # Test save_patched_file
    print("Testing save_patched_file...")
    request = {
        "jsonrpc": "2.0",
        "method": "save_patched_file",
        "params": ["C:/Users/aelna/Desktop/patched_fagality2333.dll"],
        "id": 4
    }
    
    try:
        response = requests.post(url, headers=headers, json=request)
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_save_function()