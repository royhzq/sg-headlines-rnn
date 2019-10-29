import os
import json
import unittest
from app import app

class APITests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_get_api(self):
        response = self.app.get('/headlines')
        self.assertEqual(response.status_code, 200)

    def test_post_api(self):
        data = json.dumps({
            'start_string': "Hello World",
            'n_headlines': 10,
        })
        response = self.app.post('/headlines',
                data=data,
                content_type='application/json'
            )
        n_headlines = len(json.loads(response.data))
        self.assertTrue(
            n_headlines == 10 and
            response.status_code == 200
        )
        
    def test_post_no_input(self):
        # Test for empty inputs
        data = json.dumps({
            'start_string': None,
            'n_headlines': None,
        })
        response = self.app.post('/headlines',
                data=data,
                content_type='application/json'
            )
        self.assertEqual(response.status_code, 400)

    def test_post_unknown_char(self):
        # Test for input string with unseen characters
        # in char2idx dictionary
        data = json.dumps({
            'start_string': "he",
            'n_headlines': 2,
        })
        response = self.app.post('/headlines',
                data=data,
                content_type='application/json'
            )
        n_headlines = len(json.loads(response.data))
        self.assertTrue(
            n_headlines == 2 and
            response.status_code == 200
        )        

if __name__ == "__main__":
    unittest.main()