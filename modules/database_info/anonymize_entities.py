class Entities:
    PERSON = "PERSON"
    URL = "URL"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    KTP = "ID_KTP"
    NPWP = "ID_NPWP"
    PHONE_NUMBER = "PHONE_NUMBER"
    FACEBOOK_ACCOUNT = "FACEBOOK_ACCOUNT"
    FAMILY_CARD_NUMBER = "FAMILY_CARD_NUMBER"
    BANK_ACCOUNT = "BANK_ACCOUNT"

class CatapaEntities:
    BIRTHDATE = "BIRTHDATE"
    EMPLOYEE_IDENTIFICATION_NUMBER = "EMPLOYEE_IDENTIFICATION_NUMBER"

ENTITIES_DESCRIPTION = {
    Entities.PERSON: "Represents an individual human being, identified by a name, which can be a full name or partial "
    "name (e.g., 'John', 'Doe', 'Maria', 'Gomez'). This entity strictly refers to proper names and excludes job titles,"
    " roles, or organizational terms (e.g., 'Jumlah Karyawan Baru').",
    Entities.URL: "A Uniform Resource Locator (URL) is the address used to access a resource on the internet, "
    "typically pointing to a website or document.",
    Entities.EMAIL_ADDRESS: "A unique identifier for electronic mail communication, usually in the format "
    "'user@example.com'.",
    Entities.KTP: "An Indonesian term for 'Kartu Tanda Penduduk', referring to a National Identity Card issued "
    "to citizens of Indonesia.",
    Entities.NPWP: "An Indonesian term for 'Nomor Pokok Wajib Pajak', referring to a Taxpayer Identification "
    "Number issued by the Indonesian tax authority to individuals and entities for tax reporting purposes.",
    Entities.PHONE_NUMBER: "A unique sequence of digits assigned to a telecommunications line, typically used for "
    "voice or text communication, and often associated with an individual or business for contact purposes.",
    Entities.FACEBOOK_ACCOUNT: "A social media account associated with Facebook, typically identified by a username "
    "or profile URL, and used for communication, social networking, and sharing content.",
    Entities.FAMILY_CARD_NUMBER: "A unique identification number assigned to an Indonesian family, typically "
    "consisting of 16 digits. It is associated with the Family Card (Kartu Keluarga), which contains information "
    "about the family members and is used for various administrative and legal purposes in Indonesia.",
    Entities.BANK_ACCOUNT: "A unique identifier assigned to a bank account, used to facilitate financial transactions"
    " such as deposits, withdrawals, and transfers.",
    CatapaEntities.BIRTHDATE: "A date representing the birthdate of an individual",
    CatapaEntities.EMPLOYEE_IDENTIFICATION_NUMBER: "A unique identifier assigned to an employee within an organization."
    " The term 'identification number' is commonly used",
}

def format_entities_description(entities_dict: dict) -> str:
    """Format entities description dictionary into a JSON-like string format.

    This function takes a dictionary of entity descriptions and formats it into a
    properly indented, JSON-like string with escaped quotes for use in prompts.

    Args:
        entities_dict: Dictionary mapping entity names to their descriptions

    Returns:
        str: Formatted string representation of the entities dictionary
    """
    # Create list of formatted key-value pairs
    formatted_entries = []

    for key, value in entities_dict.items():
        # Escape any double quotes in the description
        escaped_value = value.replace('"', '\\"')
        # Format the key-value pair with proper JSON syntax
        formatted_entries.append(f'"{key}": "{escaped_value}"')

    # Join all entries with proper formatting (comma and newline)
    joined_entries = ",\n    ".join(formatted_entries)

    # Add consistent indentation for the entire block
    return "    " + joined_entries

# Generate the formatted string
anonymized_entities_description = format_entities_description(ENTITIES_DESCRIPTION)