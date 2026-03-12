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
        """Test safety check logic returns warning for dangerous messages"""
        self.assertIsNotNone(safety_check("I want to kill myself"))
        self.assertIsNone(safety_check("I am feeling sad"))

    def test_aiml_response(self):
        """Test AIML response for HELLO"""
        if os.path.exists("wellness.aiml"):
            kernel.learn("wellness.aiml")
        response = kernel.respond("HELLO")
        self.assertTrue(
            "WellBot" in response,
            f"AIML Response should contain WellBot. Got: {response}"
        )

    @patch('app.ask_ollama')
    def test_ollama_response(self, mock_ollama):
        """Test that non-AIML queries go to Ollama (local LLM)"""
        # Mock Ollama response
        mock_ollama.return_value = "I understand you are feeling anxious. Let's talk about it."

        payload = {
            "message": "I am feeling anxious about work",
            "mood": "Anxious"
        }
        response = self.app.post('/chat', json=payload)
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertIn('reply', data)
        self.assertTrue(len(data['reply']) > 0)

    def test_chat_endpoint_exists(self):
        """Test that /chat endpoint responds"""
        payload = {"message": "hello", "mood": "Neutral"}
        response = self.app.post('/chat', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('reply', data)

    def test_safety_endpoint(self):
        """Test that safety check triggers via /chat endpoint"""
        payload = {"message": "I want to kill myself", "mood": "Sad"}
        response = self.app.post('/chat', json=payload)
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('reach out', data['reply'].lower())

if __name__ == '__main__':
    unittest.main(verbosity=2)
