import os
from typing import Any, Dict, List, Optional

import mysql.connector
from dotenv import load_dotenv
from mysql.connector import Error


def connect_to_mariadb(database_name: str) -> Optional[mysql.connector.connection.MySQLConnection]:
    """Establishes a connection to a MariaDB database using environment variables.

    Args:
        database_name (str): The name of the database to connect to.

    Returns:
        Optional[mysql.connector.connection.MySQLConnection]: A connection object if successful,
        None if the connection fails.

    Raises:
        Error: If there is an error during the connection attempt.
    """
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER", "app_user_demo"),
            password=os.getenv("DB_PASSWORD", "StrongPassw0rd!"),
            database=database_name
        )

        if connection.is_connected():
            db_info = connection.get_server_info()
            print(f"Connected to MariaDB Server version {db_info}")
            return connection

    except Error as e:
        print(f"Error connecting to MariaDB: {e}")
        return None

def execute_query(connection: mysql.connector.connection.MySQLConnection, query: str) -> List[Dict[str, Any]]:
    """Execute a query on the MariaDB database.

    Args:
        connection: The database connection object.
        query: The SQL query to execute.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the query results.
    """
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result
