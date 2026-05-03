import json

def clean_data(input_file, output_file):
    # Load the JSON data
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return

    original_count = len(data['results'])
    seen_tags = set()
    cleaned_results = []

    # Process results
    for entry in data['results']:
        tag = entry.get('tag')
        if tag not in seen_tags:
            cleaned_results.append(entry)
            seen_tags.add(tag)
    
    # Update the results list and the metadata count
    data['results'] = cleaned_results
    if 'meta' in data:
        data['meta']['completed'] = len(cleaned_results)

    # Save the cleaned data
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    removed_count = original_count - len(cleaned_results)
    print(f"Process Complete:")
    print(f"------------------")
    print(f"Original entries:  {original_count}")
    print(f"Valid & Unique:    {len(cleaned_results)}")
    print(f"Total Removed:     {removed_count} (duplicates)")
    print(f"Cleaned file:      {output_file}")

# --- Run ---
if __name__ == "__main__":
    input_filename = 'sweep_results.json' # change to your filename
    output_filename = 'sweep_resultsCleaned.json'
    
    clean_data(input_filename, output_filename)