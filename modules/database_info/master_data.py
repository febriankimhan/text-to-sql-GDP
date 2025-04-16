import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Define required paths
employee_schema_path = "schema/employee_table.json"
payroll_schema_path = "schema/payroll_table.json"
time_management_schema_path = "schema/time_management_table.json"

master_data_attendance_path = "master_data/attendance_statuses.csv"
master_data_employment_status_path = "master_data/employment_status_types.csv"
master_data_employment_path = "master_data/employment_types.csv"

class MasterDataLoader:
    """Class to load and process master data from CSV files and schema files.

    This class handles loading master data from CSV files and extracting master data
    information from schema files, with proper error handling and logging.
    """

    def __init__(self, master_data_paths: Dict[str, str], schema_paths: Dict[str, str]):
        """Initialize the MasterDataLoader with paths to master data and schema files.

        Args:
            master_data_paths: Dictionary mapping data types to file paths
            schema_paths: Dictionary mapping domains to schema file paths
        """
        self.master_data_paths = master_data_paths
        self.schema_paths = schema_paths
        self.master_data_dict = {}
        self.db_master_data_dict = {domain: "" for domain in schema_paths.keys()}

    def read_csv_master_data(self, file_path: str) -> Optional[List[str]]:
        """Read CSV file and return list of values from 'name' column.

        Args:
            file_path: Path to the CSV file containing master data

        Returns:
            List of values from the 'name' column or None if file not found or invalid
        """
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: Master data file not found: {file_path}")
            return None

        try:
            with path.open('r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                # Verify 'name' column exists
                if csv_reader.fieldnames is None or 'name' not in csv_reader.fieldnames:
                    print(f"Warning: CSV file {file_path} does not contain a 'name' column")
                    return None
                return [row['name'] for row in csv_reader]
        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
            return None

    @staticmethod
    def extract_schema_master_data(schema_data: Dict[str, Any]) -> str:
        """Extract master_data information from schema.

        Args:
            schema_data: Dictionary containing schema data

        Returns:
            Formatted string of master data information
        """
        master_data_info = []

        for table_name, table_info in schema_data.items():
            if isinstance(table_info, dict) and 'master_data' in table_info:
                for field, values in table_info['master_data'].items():
                    master_data_info.append(f"{table_name} - {field} = {values}")

        return "\n".join(master_data_info)

    def load_master_data(self) -> Dict[str, str]:
        """Load master data from CSV files.

        Returns:
            Dictionary mapping data types to formatted master data strings
        """
        for key, file_path in self.master_data_paths.items():
            values = self.read_csv_master_data(file_path)
            if values:  # Only add if we got values
                self.master_data_dict[key] = f"{key} - name = {values}"

        return self.master_data_dict

    def load_schema_master_data(self) -> Dict[str, str]:
        """Load master data from schema files.

        Returns:
            Dictionary mapping domains to formatted master data strings
        """
        for schema_file, domain in self.schema_paths.items():
            path = Path(schema_file)
            if not path.exists():
                print(f"Warning: Schema file not found: {schema_file}")
                continue

            try:
                with path.open('r', encoding='utf-8') as f:
                    try:
                        schema_data = json.load(f)
                        self.db_master_data_dict[domain] = self.extract_schema_master_data(schema_data)
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON in schema file: {schema_file}")
            except Exception as e:
                print(f"Error reading schema file {schema_file}: {e}")

        return self.db_master_data_dict

    def load_all(self) -> tuple[Dict[str, str], Dict[str, str]]:
        """Load all master data from both CSV and schema files.

        Returns:
            Tuple containing (master_data_dict, db_master_data_dict)
        """
        self.load_master_data()
        self.load_schema_master_data()
        return self.master_data_dict, self.db_master_data_dict

# Usage example:
# Define master data files
master_data_files = {
    'attendance_statuses': master_data_attendance_path,
    'employment_status_types': master_data_employment_status_path,
    'employment_types': master_data_employment_path
}

# Map schema files to their domains
schema_file_mapping = {
    employee_schema_path: 'employee',
    payroll_schema_path: 'payroll',
    time_management_schema_path: 'time_management'
}

# Create loader and load all data
loader = MasterDataLoader(master_data_files, schema_file_mapping)
master_data_dict, db_master_data_dict = loader.load_all()

# Show result
employee_master_data = master_data_dict['employment_status_types'] + '\n' + master_data_dict['employment_types'] + '\n' + db_master_data_dict['employee']
time_management_master_data = master_data_dict['attendance_statuses'] + '\n' + db_master_data_dict['time_management'] + '\n' + master_data_dict['employment_types']