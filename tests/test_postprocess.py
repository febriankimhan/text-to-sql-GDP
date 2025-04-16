import os
import sys
import unittest
from typing import Any, Dict, Optional

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.postprocess import extract_json_from_string, extract_sql_query


class TestExtractJsonFromString(unittest.TestCase):
    """Test cases for the extract_json_from_string function."""

    def test_json_in_code_block(self) -> None:
        """Test extracting JSON from a markdown code block."""
        data = """
        ```json
        {
          "name": "John Doe",
          "age": 30,
          "is_active": true
        }
        ```
        """
        expected = {
            "name": "John Doe",
            "age": 30,
            "is_active": True
        }
        result = extract_json_from_string(data)
        self.assertEqual(result, expected)

    def test_json_in_code_block_without_language(self) -> None:
        """Test extracting JSON from a markdown code block without language specification."""
        data = """
        ```
        {
          "name": "John Doe",
          "age": 30,
          "is_active": true
        }
        ```
        """
        expected = {
            "name": "John Doe",
            "age": 30,
            "is_active": True
        }
        result = extract_json_from_string(data)
        self.assertEqual(result, expected)

    def test_plain_json_string(self) -> None:
        """Test extracting JSON from a plain string."""
        data = """
        {
          "name": "John Doe",
          "age": 30,
          "is_active": true
        }
        """
        expected = {
            "name": "John Doe",
            "age": 30,
            "is_active": True
        }
        result = extract_json_from_string(data)
        self.assertEqual(result, expected)

    def test_json_with_surrounding_text(self) -> None:
        """Test extracting JSON from a string with surrounding text."""
        data = """
        Here is some JSON data:
        {
          "name": "John Doe",
          "age": 30,
          "is_active": true
        }
        And here is some more text.
        """
        expected = {
            "name": "John Doe",
            "age": 30,
            "is_active": True
        }
        result = extract_json_from_string(data)
        self.assertEqual(result, expected)

    def test_invalid_json(self) -> None:
        """Test handling of invalid JSON."""
        data = """
        ```json
        {
          "name": "John Doe",
          "age": 30,
          "is_active": true,
        }
        ```
        """
        result = extract_json_from_string(data)
        self.assertIsNone(result)

    def test_no_json(self) -> None:
        """Test handling of strings with no JSON."""
        data = "This is just a plain text string with no JSON."
        result = extract_json_from_string(data)
        self.assertIsNone(result)

    def test_nested_json(self) -> None:
        """Test extracting nested JSON objects."""
        data = """
        ```json
        {
          "person": {
            "name": "John Doe",
            "age": 30,
            "address": {
              "street": "123 Main St",
              "city": "Anytown"
            }
          },
          "is_active": true
        }
        ```
        """
        expected = {
            "person": {
                "name": "John Doe",
                "age": 30,
                "address": {
                    "street": "123 Main St",
                    "city": "Anytown"
                }
            },
            "is_active": True
        }
        result = extract_json_from_string(data)
        self.assertEqual(result, expected)

    def test_json_with_arrays(self) -> None:
        """Test extracting JSON with arrays."""
        data = """
        ```json
        {
          "name": "John Doe",
          "skills": ["Python", "JavaScript", "SQL"],
          "projects": [
            {"name": "Project A", "status": "completed"},
            {"name": "Project B", "status": "in-progress"}
          ]
        }
        ```
        """
        expected = {
            "name": "John Doe",
            "skills": ["Python", "JavaScript", "SQL"],
            "projects": [
                {"name": "Project A", "status": "completed"},
                {"name": "Project B", "status": "in-progress"}
            ]
        }
        result = extract_json_from_string(data)
        self.assertEqual(result, expected)


class TestExtractSqlQuery(unittest.TestCase):
    """Test cases for the extract_sql_query function."""

    def test_sql_in_json_code_block(self) -> None:
        """Test extracting SQL from a JSON object in a code block."""
        data = """
        ```json
        {
          "sql_query": "SELECT * FROM users WHERE age > 18;"
        }
        ```
        """
        expected = "SELECT * FROM users WHERE age > 18;"
        result = extract_sql_query(data)
        self.assertEqual(result, expected)

    def test_sql_in_plain_json(self) -> None:
        """Test extracting SQL from a plain JSON string."""
        data = """
        {
          "sql_query": "SELECT * FROM users WHERE age > 18;"
        }
        """
        expected = "SELECT * FROM users WHERE age > 18;"
        result = extract_sql_query(data)
        self.assertEqual(result, expected)

    def test_sql_in_sql_code_block(self) -> None:
        """Test extracting SQL from an SQL code block."""
        data = """
        ```sql
        SELECT * FROM users WHERE age > 18;
        ```
        """
        expected = "SELECT * FROM users WHERE age > 18;"
        result = extract_sql_query(data)
        self.assertEqual(result, expected)

    def test_sql_in_code_block_without_language(self) -> None:
        """Test extracting SQL from a code block without language specification."""
        data = """
        ```
        SELECT * FROM users WHERE age > 18;
        ```
        """
        expected = "SELECT * FROM users WHERE age > 18;"
        result = extract_sql_query(data)
        self.assertEqual(result, expected)

    def test_plain_sql(self) -> None:
        """Test extracting SQL from plain text."""
        data = "SELECT * FROM users WHERE age > 18;"
        expected = "SELECT * FROM users WHERE age > 18;"
        result = extract_sql_query(data)
        self.assertEqual(result, expected)

    def test_sql_with_surrounding_text(self) -> None:
        """Test extracting SQL from text with surrounding content."""
        data = """
        Here is an SQL query:
        SELECT * FROM users WHERE age > 18;
        And here is some more text.
        """
        expected = "SELECT * FROM users WHERE age > 18;"
        result = extract_sql_query(data)
        self.assertEqual(result, expected)

    def test_complex_sql_query(self) -> None:
        """Test extracting a complex SQL query with joins and subqueries."""
        data = """
        ```json
        {
          "sql_query": "SELECT u.id, u.name, COUNT(o.id) AS order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE u.created_at > '2023-01-01' GROUP BY u.id, u.name HAVING COUNT(o.id) > 5 ORDER BY order_count DESC LIMIT 10;"
        }
        ```
        """
        expected = "SELECT u.id, u.name, COUNT(o.id) AS order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE u.created_at > '2023-01-01' GROUP BY u.id, u.name HAVING COUNT(o.id) > 5 ORDER BY order_count DESC LIMIT 10;"
        result = extract_sql_query(data)
        self.assertEqual(result, expected)

    def test_sql_with_multiple_statements(self) -> None:
        """Test extracting SQL with multiple statements."""
        data = """
        ```json
        {
          "sql_query": "SELECT * FROM users; SELECT * FROM orders;"
        }
        ```
        """
        expected = "SELECT * FROM users; SELECT * FROM orders;"
        result = extract_sql_query(data)
        self.assertEqual(result, expected)

    def test_no_sql(self) -> None:
        """Test handling of strings with no SQL."""
        data = "This is just a plain text string with no SQL."
        result = extract_sql_query(data)
        self.assertIsNone(result)

    def test_sql_with_comments(self) -> None:
        """Test extracting SQL with comments."""
        data = """
        ```json
        {
          "sql_query": "-- Get all active users\nSELECT * FROM users\n-- Only include active ones\nWHERE is_active = TRUE;"
        }
        ```
        """
        expected = "-- Get all active users\nSELECT * FROM users\n-- Only include active ones\nWHERE is_active = TRUE;"
        result = extract_sql_query(data)
        self.assertEqual(result, expected)

    def test_original_example(self) -> None:
        """Test the original example from the user query."""
        data = """
        ```json
        {
          "sql_query": "SELECT DISTINCT e.id AS employee_id, CONCAT(e.name) AS employee_name, lbv.remaining AS remaining_leave_days FROM employees e JOIN leave_balances_view lbv ON e.id = lbv.employee_id WHERE YEAR(lbv.date) = 2024 AND lbv.remaining < 3;"
        }
        ```
        """
        expected = "SELECT DISTINCT e.id AS employee_id, CONCAT(e.name) AS employee_name, lbv.remaining AS remaining_leave_days FROM employees e JOIN leave_balances_view lbv ON e.id = lbv.employee_id WHERE YEAR(lbv.date) = 2024 AND lbv.remaining < 3;"
        result = extract_sql_query(data)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()