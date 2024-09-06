import json
import re

def to_space_delimited(term):
    # Step 1: Remove Salesforce-specific suffixes 
    term = re.sub(r'(__c|__ChangeEvent|__History|__Share|__e|__mdt|__hd)$', '', term)
    term = re.sub(r'\s+', ' ', term)  # Ensure single spaces

    # Step 2: Remove any prefixes
    term = re.sub(r'^[a-zA-Z0-9]+__', '', term)
    term = re.sub(r'\s+', ' ', term)  # Ensure single spaces

    # Step 3: Replace underscores with spaces
    term = term.replace('_', ' ')
    term = re.sub(r'\s+', ' ', term)  # Ensure single spaces

    # Step 4: Insert spaces between camelCase words
    term = re.sub(r'([a-z])([A-Z])', r'\1 \2', term)
    term = re.sub(r'\s+', ' ', term)  # Ensure single spaces

    # Step 5: Handle acronyms or adjacent uppercase letters
    term = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', term)
    term = re.sub(r'\s+', ' ', term)  # Ensure single spaces

    # Step 6: Insert spaces between letters and numbers
    term = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', term)
    term = re.sub(r'\s+', ' ', term)  # Ensure single spaces

    # Step 7: Convert to lowercase and strip any extra spaces
    return term.strip().lower()

def create_salesforce_object_mapping(input_file, output_file):
    # Load the Salesforce object data
    with open(input_file, 'r') as f:
        data = json.load(f)

    object_aliases = []

    # Process each object from the "object_list"
    for obj in data['object_list']: 
        cleaned_term = to_space_delimited(obj)
        object_aliases.append({
            "term": cleaned_term,
            "salesforce_object": obj 
        })

    # Create the output dictionary 
    output_data = {
        "object_aliases": object_aliases  
    }

    # Write the output to a file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

# File paths
input_file = 'salesforce_object_data.json'  # replace with actual path
output_file = 'salesforce_object_mapping.json'

# Generate the mapping
create_salesforce_object_mapping(input_file, output_file)

print(f"Mapping file '{output_file}' has been created successfully.")