import unittest
import json
import os
from unittest.mock import MagicMock, patch
from app import app, safety_check, kernel

class TestWellnessChatbot(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_safety_check(self):
        """Test safety check logic"""
        self.assertIsNotNone(safety_check("I want to kill myself"))
        self.assertIsNone(safety_check("I am feeling sad"))

    def test_aiml_response(self):
        """Test AIML response for HELLO"""
        # Ensure kernel is loaded
        if os.path.exists("wellness.aiml"):
             kernel.learn("wellness.aiml")
        
        response = kernel.respond("HELLO")
        self.assertTrue("WellBot" in response, f"AIML Response should contain WellBot. Got: {response}")

    @patch('app.model.generate_content')
    def test_gemini_fallback(self, mock_generate):
        """Test that non-AIML/non-safe queries go to Gemini"""
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = "I understand you are feeling anxious."
        mock_generate.return_value = mock_response

        payload = {
            "message": "I am feeling anxious about work",
            "mood": "Anxious"
        }
        response = self.app.post('/chat', json=payload)
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['reply'], "I understand you are feeling anxious.")

if __name__ == '__main__':
    unittest.main()
