#!/bin/bash

# Setup script for text-to-SQL project
# This script installs all dependencies required for running the notebooks in this project

set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up environment for text-to-SQL project..."

# Create required directories first - move this to the beginning to ensure they're created
# even if the script encounters issues later
echo "Creating required directories..."
mkdir -p text_to_sql/models
mkdir -p sql_generator_result
echo "Created directories: text_to_sql/models and sql_generator_result"
ls -la text_to_sql sql_generator_result

# Check if conda is installed
CONDA_PATH="/root/miniconda3/bin/conda"
if [ -f "$CONDA_PATH" ]; then
    echo "Found Conda at $CONDA_PATH"
    # Add conda to PATH if not already there
    export PATH="$PATH:/root/miniconda3/bin"
elif ! command -v conda &> /dev/null; then
    echo "Conda is not installed or not found at $CONDA_PATH. Please install Miniconda or Anaconda before running this script."
    exit 1
fi

# Create a Conda environment
CURRENT_DIR=$(pwd)
ENV_PATH="$CURRENT_DIR/text-to-sql-env"
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating Conda environment at $ENV_PATH..."
    $CONDA_PATH create --prefix $ENV_PATH python=3.11 ipykernel -y
fi

# Set up conda initialization for bash
echo "Ensuring conda is properly initialized..."
eval "$($CONDA_PATH shell.bash hook)"

# Activate the Conda environment using eval
echo "Activating Conda environment..."
eval "$($CONDA_PATH shell.bash activate $ENV_PATH)" || {
    echo "Error: Failed to activate the Conda environment."
    echo "Please try to activate it manually with:"
    echo "$CONDA_PATH activate $ENV_PATH"
    echo "Then run this script again."
    exit 1
}

# Ensure pip is up to date
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install torch==2.6.0
pip install numpy==1.26.4
pip install pandas==2.2.2
pip install tqdm==4.66.4
pip install python-dotenv==1.0.1
pip install pydantic
pip install sqlglot
pip install sqlalchemy==2.0.38
pip install pymysql==1.0.2
pip install cryptography==44.0.2
pip install pynvml
pip install ipywidgets
pip install widgetsnbextension
pip install ipympl

# Install AWS related dependencies
echo "Installing AWS related dependencies..."
pip install boto3==1.35.71
pip install -q langchain-aws==0.2.7
# Install AWS CLI
echo "Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -o awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws

# Install Transformers and related libraries
echo "Installing Transformers and related libraries..."
pip install huggingface-hub==0.29.1
pip install unsloth unsloth_zoo
pip install transformers==4.49.0

# Install LangChain and related libraries
echo "Installing LangChain and related libraries..."
pip install langchain==0.3.0
pip install langchain_openai==0.2.14
pip install langchain-community

# Install Google Sheets integration
echo "Installing Google Sheets integration..."
pip install google-api-python-client==2.100.0
pip install gspread==5.10.0

# Install MariaDB/MySQL connector
echo "Installing database connectors..."
pip install mysql-connector-python==9.2.0

# Install Jupyter support
echo "Installing Jupyter support..."
pip install jupyter
python -m ipykernel install --user --name=myenv-text-to-sql --display-name="Python 3.11 (text-to-sql)"

# Create a .env file template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo ".env file created from .env.example template. Please fill in the required credentials."
    else
        echo "Warning: .env.example not found. Creating a basic .env template..."
        cat > .env << EOF
# Google Sheets API Credentials
GOOGLE_SHEETS_CLIENT_EMAIL=
GOOGLE_SHEETS_PRIVATE_KEY=

# Database Connection Parameters
DB_HOST=
DB_USER=
DB_PASSWORD=
DB_DATABASE=
DB_PORT=3306

# Hugging Face API Token
HF_TOKEN=
EOF
        echo "Basic .env template created. Please fill in the required credentials."
    fi
fi

# Verify that required directories exist at the end
echo "Verifying required directories..."
if [ ! -d "text_to_sql/models" ]; then
    echo "Warning: text_to_sql/models directory was not created successfully. Creating it again..."
    mkdir -p text_to_sql/models
fi

if [ ! -d "sql_generator_result" ]; then
    echo "Warning: sql_generator_result directory was not created successfully. Creating it again..."
    mkdir -p sql_generator_result
fi

echo "Setup complete! Use '$CONDA_PATH activate $ENV_PATH' to activate the Conda environment."
echo "Please fill in the credentials in the .env file before running the notebooks."
echo ""
echo "If you encounter errors while activating, try the following commands in a terminal:"
echo "$CONDA_PATH init bash"
echo "source ~/.bashrc"