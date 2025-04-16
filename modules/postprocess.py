import json
import re
from typing import Any, Dict, Optional


def extract_json_from_string(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from a string that may contain markdown code blocks.

    This function handles strings that may have JSON embedded within markdown code blocks
    (```json ... ```) or plain JSON strings.

    Args:
        text: The input string that may contain JSON data, possibly within markdown code blocks.

    Returns:
        Dict[str, Any]: The extracted JSON data as a Python dictionary.
        None: If no valid JSON could be extracted.
    """
    # Try to extract JSON from markdown code blocks first
    json_block_pattern = r"```(?:json)?\s*\n([\s\S]*?)\n\s*```"
    matches = re.findall(json_block_pattern, text)

    if matches:
        # Try each matched block until we find valid JSON
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # If no code blocks or no valid JSON in code blocks, try the whole string
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # As a last resort, try to find anything that looks like a JSON object
        json_pattern = r"\{[\s\S]*\}"
        potential_json = re.search(json_pattern, text)
        if potential_json:
            try:
                return json.loads(potential_json.group(0))
            except json.JSONDecodeError:
                return None

    return None


def extract_sql_query(text: str) -> Optional[str]:
    """Extract a SQL query from a string that may contain JSON or markdown code blocks.

    This function first attempts to extract JSON from the input string, then looks for
    a 'sql_query' field in the extracted JSON. If that fails, it tries to extract any SQL
    query directly from the text.

    Args:
        text: The input string that may contain a SQL query, possibly within JSON or code blocks.

    Returns:
        str: The extracted SQL query.
        None: If no SQL query could be extracted.
    """
    # First try to extract as JSON
    try:
        json_data = extract_json_from_string(text)

        if json_data and "sql_query" in json_data:
            return json_data["sql_query"]

        # If that fails, try to extract SQL directly from code blocks
        sql_block_pattern = r"```(?:sql)?\s*\n([\s\S]*?)\n\s*```"
        matches = re.findall(sql_block_pattern, text)

        if matches:
            return matches[0].strip()

        # As a last resort, look for anything that looks like a SQL query
        # (This is a simplified approach and might not work for all cases)
        sql_pattern = r"SELECT[\s\S]*?(?:;|$)"
        potential_sql = re.search(sql_pattern, text, re.IGNORECASE)
        if potential_sql:
            return potential_sql.group(0).strip()
    except Exception as e:
        print(f"Error extracting SQL query: {e}")
        return None

    return None


def extract_clean_sql_query(text: str) -> Optional[str]:
    """Extract a SQL query from text and remove any explanations that follow it.

    This function extracts a SQL query and removes any explanations, comments, or
    additional text that might follow the actual query. It handles various formats
    including queries ending with semicolons or queries followed by explanations.

    Args:
        text: The input string containing a SQL query possibly followed by explanations.

    Returns:
        str: The cleaned SQL query with explanations removed.
        None: If no SQL query could be extracted.
    """
    # First try to get any SQL query using the existing function
    query = extract_sql_query(text)

    if not query:
        return None

    # If the query ends with a semicolon, that's a clear boundary
    if ";" in query:
        return query.split(";")[0].strip() + ";"

    # Look for common explanation markers
    explanation_markers = [
        "\n\nExplanation:",
        "\n\nNote:",
        "\nExplanation:",
        "\nNote:",
        "\n\n1.",
        "\n1."
    ]

    for marker in explanation_markers:
        if marker in query:
            return query.split(marker)[0].strip()

    # If no clear markers, try to detect the end of the SQL query
    # SQL queries typically end with GROUP BY, ORDER BY, LIMIT, etc.
    # or have empty lines before explanations begin
    lines = query.split("\n")
    clean_lines = []

    for i, line in enumerate(lines):
        clean_lines.append(line)
        # If we find an empty line followed by text that doesn't look like SQL
        if (line.strip() == "" and i < len(lines) - 1 and
            not any(sql_keyword in lines[i+1].upper() for sql_keyword in
                   ["SELECT", "FROM", "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "JOIN", "UNION"])):
            break

    return "\n".join(clean_lines).strip()


# Example usage
if __name__ == "__main__":
    # Example with JSON in code block
    data = """
    ```json
    {
      "sql_query": "SELECT DISTINCT e.id AS employee_id, CONCAT(e.name) AS employee_name, lbv.remaining AS remaining_leave_days FROM employees e JOIN leave_balances_view lbv ON e.id = lbv.employee_id WHERE YEAR(lbv.date) = 2024 AND lbv.remaining < 3;"
    }
    ```
    """

    # Extract the SQL query
    query = extract_sql_query(data)
    print(query)