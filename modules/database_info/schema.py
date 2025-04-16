import json

# Define required paths


def format_column_info(column: dict) -> str:
    """Format column information with type and description.

    Args:
        column (dict): Dictionary containing column information with at least a 'name' key.
            May also contain 'type' and 'description' keys.

    Returns:
        str: Formatted column information string.
    """
    parts = [column['name']]

    # Add type if present
    if 'type' in column:
        parts.append(f"[{column['type']}]")

    # Add description if present
    if 'description' in column:
        parts.append(f"({column['description']})")

    return " ".join(parts)

def create_schema_dictionary(schema_data: dict) -> str:
    """Create formatted schema information for all tables.

    Args:
        schema_data (dict): Dictionary containing table schema information.
            Each key is a table name, and each value is a dictionary with table information.

    Returns:
        str: Formatted schema information for all tables.
    """
    formatted_schemas = []

    for table_name, table_info in schema_data.items():
        # Get columns with a default empty list if not present
        columns = table_info.get('columns', [])

        # Format column information using list comprehension
        column_info = [f"- {format_column_info(column)}" for column in columns]

        # Join column information with newlines
        formatted_schemas.append(f"{table_name}:\n" + "\n".join(column_info))

    return "\n\n".join(formatted_schemas)


# Initialize the main dictionary with three domains
table_schema = {domain: "" for domain in ['employee', 'payroll', 'time_management']}

employee_schema_path = "schema/employee_table.json"
payroll_schema_path = "schema/payroll_table.json"
time_management_schema_path = "schema/time_management_table.json"

# Map files to their domains
file_domain_mapping = {
    employee_schema_path: 'employee',
    payroll_schema_path: 'payroll',
    time_management_schema_path: 'time_management'
}

# Process each schema file
for schema_file, domain in file_domain_mapping.items():
    try:
        with open(schema_file) as f:
            schema_data = json.load(f)
            table_schema[domain] = create_schema_dictionary(schema_data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error processing schema file {schema_file}: {e}")

## Show Schema
employee_schema = table_schema['employee'] + '\n' + table_schema['payroll']
time_management_schema = table_schema['time_management']