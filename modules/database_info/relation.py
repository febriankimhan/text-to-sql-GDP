import json


def create_relations_dictionary(schema_data: dict) -> str:
    """Create formatted relations information for all tables.

    Args:
        schema_data (dict): Dictionary containing table schema information.
            Each key is a table name, and each value is a dictionary with table information.

    Returns:
        str: Formatted relations information for all tables.
    """
    formatted_relations = []

    for table_name, table_info in schema_data.items():
        # Skip if no foreign keys
        if 'foreign_keys' not in table_info or not table_info['foreign_keys']:
            continue

        # Start with table name
        relations = [f"{table_name}:"]

        # Add each foreign key relation
        relations.extend([
            f"- {fk['ref_table']} referenced by {fk['column']}"
            for fk in table_info['foreign_keys']
        ])

        formatted_relations.append("\n".join(relations))

    return "\n".join(formatted_relations)

# Initialize the relations dictionary for all domains
domains = ['employee', 'payroll', 'time_management']
table_relations = {domain: "" for domain in domains}

employee_schema_path = "schema/employee_table.json"
payroll_schema_path = "schema/payroll_table.json"
time_management_schema_path = "schema/time_management_table.json"

# Map schema files to their domains
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
            table_relations[domain] = create_relations_dictionary(schema_data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error processing relations in {schema_file}: {e}")

# show relations
employee_relations = table_relations['employee'] + '\n' + table_relations['payroll']
time_management_relations = table_relations['time_management']