"""SQL query testing module."""
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class SQLQueryTester:
    """Class to handle SQL query testing across different databases."""

    def __init__(self, databases: Dict[str, Session], default_db: str = 'core', refresh_interval: Optional[int] = None):
        """Initialize with database sessions.

        Args:
            databases (Dict[str, Session]): Dictionary mapping database names to their sessions
            default_db (str): Name of the default database to use when not specified
            refresh_interval (Optional[int]): Seconds between filter refreshes. None means no auto-refresh.
        """
        self.databases = databases
        self.default_db = default_db
        self.refresh_interval = refresh_interval
        self.last_refresh = datetime.now()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Cache for prepared queries and filters
        self._query_cache = {}
        self.filters = self._get_employment_filters()


    def _get_distinct_values(self, db: Session, table: str, column: str) -> List[str]:
        """Get distinct values from a specified column in a table.

        Args:
            db (Session): Database session
            table (str): Table name
            column (str): Column name

        Returns:
            List[str]: List of distinct non-null values
        """
        query = f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL"
        cursor = db.cursor(dictionary=True)
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return [str(row[column]) for row in result]

    def _get_filters(self, db_name: str, filters_config: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Get filter values based on provided configuration.

        Args:
            db_name (str): Name of the database to query (e.g., 'core', 'time')
            filters_config (List[Dict[str, str]]): List of filter configurations.
                Each config should have:
                - 'key': Key for the returned dictionary
                - 'table': Table name to query
                - 'column': Column name to get distinct values from

        Returns:
            Dict[str, List[str]]: Dictionary containing filter values mapped to their keys
        """
        db = self.databases.get(db_name)
        if not db:
            raise ValueError(f"Database '{db_name}' not found")

        result = {}
        for config in filters_config:
            values = self._get_distinct_values(db, config['table'], config['column'])
            if not values:
                self.logger.warning(f"No values found for {config['table']}.{config['column']}")
            result[config['key']] = values

        return result

    def _get_employment_filters(self) -> Dict[str, List[str]]:
        """Get all filter values needed for employment status queries.

        Returns:
            Dict[str, List[str]]: Dictionary containing filter values
        """
        # Define the filter configuration
        filters_config = [
            {'key': 'organization_id', 'table': 'employment_statuses', 'column': 'organization_id'},
            {'key': 'job_level_id', 'table': 'employment_statuses', 'column': 'job_level_id'},
            {'key': 'location_id', 'table': 'employment_statuses', 'column': 'location_id'}
        ]

        return self._get_filters(self.default_db, filters_config)

    def refresh_filters_if_needed(self) -> None:
        """Refresh filters if the refresh interval has passed."""
        if not self.refresh_interval:
            return

        now = datetime.now()
        elapsed = (now - self.last_refresh).total_seconds()

        if elapsed >= self.refresh_interval:
            self.logger.info("Refreshing filters due to interval expiration")
            self.filters = self._get_employment_filters()
            self.last_refresh = now
            # Clear cache when filters change
            self._query_cache.clear()

    def prepare_query(self, sql_query: str) -> str:
        """Replace placeholder values in SQL query with actual filter values.

        Args:
            sql_query (str): Original SQL query with placeholders like [ORGANIZATION_IDS]
                or '[ORGANIZATION_IDS]'

        Returns:
            str: SQL query with placeholders replaced by actual filter values
        """
        # Input validation
        if not sql_query:
            raise ValueError("SQL query cannot be empty")

        # Check cache first
        if sql_query in self._query_cache:
            return self._query_cache[sql_query]

        # Make sure filters are up-to-date
        self.refresh_filters_if_needed()
        # Create filter values with proper SQL formatting
        filter_values = {
            'organization_id': ", ".join(f"'{id}'" for id in self.filters['organization_id']),
            'job_level_id': ", ".join(f"'{id}'" for id in self.filters['job_level_id']),
            'location_id': ", ".join(f"'{id}'" for id in self.filters['location_id'])
        }

        # Handle both quoted and unquoted placeholders
        replacements = {
            # For quoted placeholders: '([ORGANIZATION_IDS])'
            "'[ORGANIZATION_IDS]'": filter_values['organization_id'],
            "'[JOB_LEVEL_IDS]'": filter_values['job_level_id'],
            "'[LOCATION_IDS]'": filter_values['location_id'],

            # For unquoted placeholders: ([ORGANIZATION_IDS])
            "[ORGANIZATION_IDS]": filter_values['organization_id'],
            "[JOB_LEVEL_IDS]": filter_values['job_level_id'],
            "[LOCATION_IDS]": filter_values['location_id']
        }

        # Replace all placeholders
        prepared_query = sql_query
        for placeholder, value in replacements.items():
            prepared_query = prepared_query.replace(placeholder, value)

        # Store in cache
        self._query_cache[sql_query] = prepared_query

        return prepared_query

    def format_query_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format query results by converting special data types to standard formats.

        Args:
            results (List[Dict[str, Any]]): List of dictionaries containing query results

        Returns:
            List[Dict[str, Any]]: Formatted results with dates as strings and decimals as integers
        """
        formatted_results = []
        for row in results:
            formatted_row = {}
            for key, value in row.items():
                # Check if value has a strftime method (date/datetime objects)
                if hasattr(value, 'strftime'):
                    formatted_row[key] = value.strftime('%Y-%m-%d')
                # Check if value is a Decimal
                elif str(type(value)).find('Decimal') > -1:
                    # Convert to int if it's a whole number, otherwise to float
                    try:
                        if value % 1 == 0:
                            formatted_row[key] = int(value)
                        else:
                            formatted_row[key] = float(value)
                    except:
                        formatted_row[key] = float(value)
                else:
                    formatted_row[key] = value
            formatted_results.append(formatted_row)

        return formatted_results

    def execute_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query on specified database.

        Args:
            sql_query (str): SQL query to execute
            db_name (str, optional): Name of database to use. If None, uses the default database.

        Returns:
            List[Dict[str, Any]]: Query results as list of dictionaries with formatted values
        """
        db_name = self.default_db

        connection = self.databases.get(db_name)
        if not connection:
            raise ValueError(f"Database '{db_name}' not found")
        cursor = connection.cursor(dictionary=True)
        try:
            # Prepare and execute query
            final_query = self.prepare_query(sql_query)
            cursor.execute(final_query)
            results = cursor.fetchall()
            # Format results to handle special data types
            cursor.close()
            return self.format_query_results(results), []
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            return None, self.extract_error_message(error_msg)

    def extract_error_message(self, error_text: str) -> str:
        """Extract the error message from SQL error text.

        Args:
            error_text (str): The full error text containing SQL error information

        Returns:
            str: The extracted error message sentence
        """
        # Find the first line which contains the error message
        lines = error_text.strip().split('\n')
        error_line = lines[0] if lines else ""

        # If the error is about a table not existing, modify it to match the desired format
        if "Table" in error_line and "doesn't exist" in error_line:
            # Extract the table name from the original error
            import re
            table_match = re.search(r"Table '([^']+)\.([^']+)' doesn't exist", error_line)

            if table_match:
                database = table_match.group(1)
                # Replace with 'families' as requested
                return f"Error executing query: (pymysql.err.ProgrammingError) (1146, \"Table '{database}.families' doesn't exist\")"

        return error_line