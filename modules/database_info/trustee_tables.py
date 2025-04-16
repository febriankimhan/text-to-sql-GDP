import json
from pathlib import Path
from typing import Dict, Optional


def remove_redundant_trustee(trustee_text: str) -> str:
    """Remove redundant entries from relations text.

    Args:
        relations_text (str): String containing relations data with potential duplicates

    Returns:
        str: Cleaned relations text with duplicates removed
    """
    # Split the text into lines
    lines = trustee_text.strip().split('\n')

    # Use a set to track unique lines
    unique_lines = []
    seen = set()

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Add to our result if we haven't seen this line before
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    # Join the unique lines back together
    return '\n'.join(unique_lines)



def extract_data_trustee(schema_data: Dict) -> str:
    """Extract data_trustee information from schema.

    Args:
        schema_data: Dictionary containing table schema information

    Returns:
        str: Formatted string of data trustee information with each entry on a new line
    """
    trustee_info = []

    for table_name, table_info in schema_data.items():
        if isinstance(table_info, dict) and 'data_trustee' in table_info:
            trustee_info.append(f"{table_name}: {table_info['data_trustee']}")

    return "\n".join(trustee_info)

def load_data_trustee_info(schema_paths: Dict[str, str]) -> Dict[str, str]:
    """Load data trustee information from schema files.

    Args:
        schema_paths: Dictionary mapping schema file paths to domain names

    Returns:
        Dict[str, str]: Dictionary mapping domains to formatted data trustee information
    """
    # Initialize the data trustee dictionary with empty strings for each domain
    data_trustee_dict = {domain: "" for domain in set(schema_paths.values())}

    # Process each schema file
    for schema_file, domain in schema_paths.items():
        path = Path(schema_file)
        if not path.exists():
            print(f"Warning: Schema file not found: {schema_file}")
            continue

        try:
            with path.open('r', encoding='utf-8') as f:
                schema_data = json.load(f)
                data_trustee_dict[domain] = extract_data_trustee(schema_data)
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in schema file: {schema_file}")
        except Exception as e:
            print(f"Error processing schema file {schema_file}: {e}")

    return data_trustee_dict

employee_schema_path = "schema/employee_table.json"
payroll_schema_path = "schema/payroll_table.json"
time_management_schema_path = "schema/time_management_table.json"

# Map files to their domains
file_domain_mapping = {
    employee_schema_path: 'employee',
    payroll_schema_path: 'payroll',
    time_management_schema_path: 'time_management'
}

# Load data trustee information
data_trustee_dict = load_data_trustee_info(file_domain_mapping)

# Example usage:
data_trustee_employee = remove_redundant_trustee(data_trustee_dict['employee'] + '\n' + data_trustee_dict['payroll'])
data_trustee_time_management = data_trustee_dict['time_management']