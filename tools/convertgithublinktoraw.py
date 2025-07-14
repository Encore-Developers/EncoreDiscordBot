import re

def convert_to_raw_link(github_url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)"
    
    match = re.match(pattern, github_url)
    
    if match:
        user, repo, branch, file_path = match.groups()
        
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path}"
        return raw_url
    
    pattern_alt = r"https://github\.com/([^/]+)/([^/]+)/raw/refs/heads/([^/]+)/(.+)"
    match_alt = re.match(pattern_alt, github_url)

    if match_alt:
        user, repo, branch, file_path = match_alt.groups()
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path}"
        return raw_url
    
    return None

if __name__ == "__main__":
    print("GitHub URL to Raw Download Link Converter")
    print("="*40)
    
    try:
        while True:
            input_url = input("Enter a GitHub file URL (or 'exit' to quit): ")

            if input_url.lower() == 'exit':
                print("Exiting the program. Goodbye!")
                break

            raw_download_link = convert_to_raw_link(input_url)

            if raw_download_link:
                print("\n--- Conversion Successful ---")
                print(f"Original URL: {input_url}")
                print(f"Raw Link:     {raw_download_link}")
                print("-----------------------------\n")
            else:
                print("\n[!] Invalid URL format.")
                print("Please enter a valid GitHub file URL, like:")
                print("https://github.com/user/repository/blob/main/file.txt\n")

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")

