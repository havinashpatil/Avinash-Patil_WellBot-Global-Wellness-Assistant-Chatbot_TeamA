import requests
import time

BASE_URL = "http://localhost:5000"

def test_signup():
    print("Testing /signup...")
    data = {
        "name": "Test User",
        "email": f"test_{int(time.time())}@example.com",
        "password": "testpassword",
        "language": "English"
    }
    response = requests.post(f"{BASE_URL}/signup", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()

def test_login(email, password):
    print(f"Testing /login for {email}...")
    data = {
        "email": email,
        "password": password
    }
    response = requests.post(f"{BASE_URL}/login", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()

if __name__ == "__main__":
    try:
        signup_result = test_signup()
        if signup_result.get("success"):
            email = signup_result.get("email", "test@example.com") # Note: /signup doesn't return email in original code, let's use the one we generated if possible, but the script above is static enough.
            # Use the generated email
            # Actually, I'll just use a temp email for the test
            temp_email = signup_result.get("email") # Wait, /signup returns {"success": True, "token": token, "name": data['name']}
            # I should have returned the email too, but I'll manually track it.
            pass
        
        # Test a known one
        email = f"test_{int(time.time())}@example.com"
        requests.post(f"{BASE_URL}/signup", json={
            "name": "Manual User",
            "email": email,
            "password": "mypassword",
            "language": "English"
        })
        
        test_login(email, "mypassword")
        test_login(email, "wrongpassword")
        
    except Exception as e:
        print(f"Error during verification: {e}")
        print("Make sure the Flask server is running at http://localhost:5000")
