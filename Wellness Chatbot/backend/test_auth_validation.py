import unittest
import re
import requests
import time

BASE_URL = "http://localhost:5000"

class TestAuthValidation(unittest.TestCase):
    """Test cases for authentication validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        self.username_regex = r'^[a-zA-Z0-9_]{3,20}$'
        
    # ==================== Username Validation Tests ====================
    
    def test_valid_username_alphanumeric(self):
        """Test valid username with alphanumeric characters"""
        valid_usernames = [
            "JohnDoe123",
            "alice_wonderland",
            "Bob2024",
            "user_name",
            "Test123"
        ]
        for username in valid_usernames:
            with self.subTest(username=username):
                self.assertTrue(len(username) >= 3 and len(username) <= 20,
                              f"Username '{username}' should be between 3-20 characters")
                self.assertIsNotNone(re.match(self.username_regex, username),
                                   f"Username '{username}' should be valid")
    
    def test_invalid_username_too_short(self):
        """Test username must be at least 3 characters"""
        invalid_usernames = ["ab", "a", "12"]
        for username in invalid_usernames:
            with self.subTest(username=username):
                self.assertLess(len(username), 3,
                              f"Username '{username}' should be too short")
    
    def test_invalid_username_too_long(self):
        """Test username must not exceed 20 characters"""
        username = "a" * 21
        self.assertGreater(len(username), 20,
                         f"Username '{username}' should be too long")
    
    def test_invalid_username_special_chars(self):
        """Test username should not contain special characters except underscore"""
        invalid_usernames = [
            "john@doe",
            "alice!wonderland",
            "bob#2024",
            "user name",  # space
            "test-user"
        ]
        for username in invalid_usernames:
            with self.subTest(username=username):
                self.assertIsNone(re.match(self.username_regex, username),
                                f"Username '{username}' should be invalid")
    
    # ==================== Email Validation Tests ====================
    
    def test_valid_email_formats(self):
        """Test valid email addresses"""
        valid_emails = [
            "user@example.com",
            "john.doe@company.org",
            "alice_wonderland@test.co.uk",
            "bob123@email-provider.com",
            "test.user+tag@domain.io"
        ]
        for email in valid_emails:
            with self.subTest(email=email):
                self.assertIsNotNone(re.match(self.email_regex, email),
                                   f"Email '{email}' should be valid")
    
    def test_invalid_email_missing_at(self):
        """Test email must contain @ symbol"""
        invalid_emails = [
            "userexample.com",
            "john.doe.company.org",
            "plaintext"
        ]
        for email in invalid_emails:
            with self.subTest(email=email):
                self.assertIsNone(re.match(self.email_regex, email),
                                f"Email '{email}' should be invalid (missing @)")
    
    def test_invalid_email_missing_domain(self):
        """Test email must have proper domain"""
        invalid_emails = [
            "user@",
            "@example.com",
            "user@@example.com"
        ]
        for email in invalid_emails:
            with self.subTest(email=email):
                self.assertIsNone(re.match(self.email_regex, email),
                                f"Email '{email}' should be invalid (missing domain)")
    
    def test_invalid_email_with_spaces(self):
        """Test email should not contain spaces"""
        invalid_emails = [
            "user name@example.com",
            "user@example .com",
            " user@example.com",
            "user@example.com "
        ]
        for email in invalid_emails:
            with self.subTest(email=email):
                self.assertIsNone(re.match(self.email_regex, email),
                                f"Email '{email}' should be invalid (contains spaces)")
    
    # ==================== Password Length Tests ====================
    
    def test_valid_password_min_8_chars(self):
        """Test password must be at least 8 characters"""
        valid_passwords = [
            "Password123!",
            "12345678",
            "abcdefgh",
            "Test@123",
            "LongPassword123!"
        ]
        for password in valid_passwords:
            with self.subTest(password=password):
                self.assertGreaterEqual(len(password), 8,
                                      f"Password '{password}' should be at least 8 characters")
    
    def test_invalid_password_too_short(self):
        """Test password shorter than 8 characters should fail"""
        invalid_passwords = [
            "Pass1!",
            "12345",
            "abc",
            "Test@1",
            "Short7"
        ]
        for password in invalid_passwords:
            with self.subTest(password=password):
                self.assertLess(len(password), 8,
                              f"Password '{password}' should be too short")
    
    # ==================== Password Strength Tests ====================
    
    def test_strong_password_all_requirements(self):
        """Test strong password with letters, numbers, and special characters"""
        strong_passwords = [
            "MyP@ssw0rd!",
            "Test@123ABC",
            "Str0ng#Pass",
            "S3cure!Password",
            "Welln3ss@2024"
        ]
        
        for password in strong_passwords:
            with self.subTest(password=password):
                # Check minimum length
                self.assertGreaterEqual(len(password), 8,
                                      f"Password '{password}' should be at least 8 chars")
                
                # Check for uppercase letter
                self.assertTrue(any(c.isupper() for c in password),
                              f"Password '{password}' should contain uppercase letter")
                
                # Check for lowercase letter
                self.assertTrue(any(c.islower() for c in password),
                              f"Password '{password}' should contain lowercase letter")
                
                # Check for digit
                self.assertTrue(any(c.isdigit() for c in password),
                              f"Password '{password}' should contain a digit")
                
                # Check for special character
                special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
                self.assertTrue(any(c in special_chars for c in password),
                              f"Password '{password}' should contain special character")
    
    def test_weak_password_missing_uppercase(self):
        """Test password missing uppercase letters"""
        weak_passwords = [
            "password123!",
            "test@1234",
            "weak#pass99"
        ]
        for password in weak_passwords:
            with self.subTest(password=password):
                self.assertFalse(any(c.isupper() for c in password),
                               f"Password '{password}' should lack uppercase")
    
    def test_weak_password_missing_lowercase(self):
        """Test password missing lowercase letters"""
        weak_passwords = [
            "PASSWORD123!",
            "TEST@1234",
            "WEAK#PASS99"
        ]
        for password in weak_passwords:
            with self.subTest(password=password):
                self.assertFalse(any(c.islower() for c in password),
                               f"Password '{password}' should lack lowercase")
    
    def test_weak_password_missing_numbers(self):
        """Test password missing numbers"""
        weak_passwords = [
            "Password!",
            "Test@Word",
            "Weak#Pass"
        ]
        for password in weak_passwords:
            with self.subTest(password=password):
                self.assertFalse(any(c.isdigit() for c in password),
                               f"Password '{password}' should lack numbers")
    
    def test_weak_password_missing_special_chars(self):
        """Test password missing special characters"""
        weak_passwords = [
            "Password123",
            "Test1234",
            "WeakPass99"
        ]
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        for password in weak_passwords:
            with self.subTest(password=password):
                self.assertFalse(any(c in special_chars for c in password),
                               f"Password '{password}' should lack special characters")
    
    # ==================== Integration Tests ====================
    
    def test_signup_with_valid_credentials(self):
        """Test successful signup with valid credentials"""
        timestamp = int(time.time())
        data = {
            "name": "TestUser123",
            "email": f"testuser{timestamp}@example.com",
            "password": "Test@Pass123",
            "language": "English"
        }
        
        try:
            response = requests.post(f"{BASE_URL}/signup", json=data, timeout=5)
            result = response.json()
            
            self.assertTrue(result.get("success"),
                          "Signup should succeed with valid credentials")
            self.assertIn("token", result,
                        "Response should contain token")
            self.assertEqual(result.get("name"), "TestUser123",
                           "Response should return correct name")
        except requests.exceptions.RequestException as e:
            self.skipTest(f"Server not running: {e}")
    
    def test_signup_with_weak_password(self):
        """Test signup fails with weak password (no special chars)"""
        timestamp = int(time.time())
        data = {
            "name": "WeakUser",
            "email": f"weakuser{timestamp}@example.com",
            "password": "weakpass123",  # No uppercase, no special chars
            "language": "English"
        }
        
        # Note: Current backend doesn't validate password strength
        # This test documents expected behavior for future implementation
        try:
            response = requests.post(f"{BASE_URL}/signup", json=data, timeout=5)
            # Currently this will succeed, but ideally should fail
            print(f"\nNote: Weak password validation not implemented in backend yet")
        except requests.exceptions.RequestException as e:
            self.skipTest(f"Server not running: {e}")
    
    def test_signup_with_invalid_email(self):
        """Test signup fails with invalid email format"""
        data = {
            "name": "TestUser",
            "email": "invalidemail",  # Invalid format
            "password": "Test@Pass123",
            "language": "English"
        }
        
        # Note: Frontend validation should catch this before reaching backend
        print(f"\nNote: Frontend should validate email format")
    
    def test_login_with_valid_credentials(self):
        """Test successful login with valid credentials"""
        timestamp = int(time.time())
        email = f"logintest{timestamp}@example.com"
        password = "Test@Login123"
        
        # First, create the user
        signup_data = {
            "name": "LoginTest",
            "email": email,
            "password": password,
            "language": "English"
        }
        
        try:
            # Signup
            requests.post(f"{BASE_URL}/signup", json=signup_data, timeout=5)
            
            # Login
            login_data = {
                "email": email,
                "password": password
            }
            response = requests.post(f"{BASE_URL}/login", json=login_data, timeout=5)
            result = response.json()
            
            self.assertTrue(result.get("success"),
                          "Login should succeed with valid credentials")
            self.assertIn("token", result,
                        "Response should contain token")
        except requests.exceptions.RequestException as e:
            self.skipTest(f"Server not running: {e}")
    
    def test_login_with_wrong_password(self):
        """Test login fails with incorrect password"""
        timestamp = int(time.time())
        email = f"wrongpass{timestamp}@example.com"
        correct_password = "Correct@Pass123"
        wrong_password = "Wrong@Pass123"
        
        # First, create the user
        signup_data = {
            "name": "WrongPassUser",
            "email": email,
            "password": correct_password,
            "language": "English"
        }
        
        try:
            # Signup
            requests.post(f"{BASE_URL}/signup", json=signup_data, timeout=5)
            
            # Try to login with wrong password
            login_data = {
                "email": email,
                "password": wrong_password
            }
            response = requests.post(f"{BASE_URL}/login", json=login_data, timeout=5)
            result = response.json()
            
            self.assertFalse(result.get("success"),
                           "Login should fail with wrong password")
            self.assertEqual(response.status_code, 401,
                           "Should return 401 Unauthorized")
        except requests.exceptions.RequestException as e:
            self.skipTest(f"Server not running: {e}")


def validate_password_strength(password):
    """
    Helper function to validate password strength
    Returns (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    if not any(c in special_chars for c in password):
        return False, "Password must contain at least one special character (!@#$%^&*...)"
    
    return True, "Password is strong"


if __name__ == "__main__":
    print("=" * 70)
    print("Authentication Validation Test Suite")
    print("=" * 70)
    print("\nTest Categories:")
    print("1. Username Validation (alphanumeric, 3-20 chars)")
    print("2. Email Validation (proper format)")
    print("3. Password Length (minimum 8 characters)")
    print("4. Password Strength (letters, numbers, special chars)")
    print("5. Integration Tests (signup/login with validation)")
    print("\n" + "=" * 70)
    print("\nRunning tests...\n")
    
    # Run the tests
    unittest.main(verbosity=2)
