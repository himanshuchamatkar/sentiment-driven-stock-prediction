# main.py

import os
import urllib.parse
from upstox_client.rest import ApiException
from upstox_client import Configuration, ApiClient, LoginApi
from config.keys import API_KEY, API_SECRET, REDIRECT_URI

# --- Global Access Token Storage (Will be updated after login) ---
ACCESS_TOKEN = None

# -----------------------------------------------------------------
# 1. GENERATE LOGIN URL (MANUAL INTERVENTION REQUIRED)
# -----------------------------------------------------------------
def generate_login_url():
    """Generates the URL the user needs to visit to get the authorization code."""
    auth_api = LoginApi()
    
    # Construct the base URL for authorization
    query_params = {
        'client_id': API_KEY,
        'redirect_uri': REDIRECT_URI,
        'response_type': 'code'
    }
    
    # Upstox V2 Authorization Endpoint
    auth_url = "https://api.upstox.com/v2/login/authorization/dialog"
    
    # Combine the URL and parameters
    login_url = f"{auth_url}?{urllib.parse.urlencode(query_params)}"
    
    print("\n" + "="*50)
    print("STEP 1: LOGIN REQUIRED")
    print("="*50)
    print("1. Visit the following URL in your web browser:")
    print(f"\n{login_url}\n")
    print("2. Log in using your Upstox credentials.")
    print("3. After successful login, you will be redirected to the REDIRECT_URI.")
    print("4. **Copy the ENTIRE URL from your browser's address bar.**")
    print("5. Paste the entire URL below and press Enter.")
    
    return input("Paste the final redirected URL here: ")


# -----------------------------------------------------------------
# 2. EXCHANGE CODE FOR ACCESS TOKEN
# -----------------------------------------------------------------
def get_access_token(full_redirect_url):
    """Parses the URL for the code and exchanges it for a token."""
    global ACCESS_TOKEN
    
    # Parse the URL to extract the authorization code
    parsed_url = urllib.parse.urlparse(full_redirect_url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    
    if 'code' not in query_params:
        print("\n[ERROR] Authorization Code not found in the URL. Did you copy the full, final URL correctly?")
        return None

    auth_code = query_params['code'][0]
    
    # Prepare the token request
    token_api = LoginApi()
    
    try:
        # Note: Upstox uses the client_id/secret for authentication directly in the token request body
        response = token_api.token(
            code=auth_code,
            client_id=API_KEY,
            client_secret=API_SECRET,
            redirect_uri=REDIRECT_URI,
            grant_type='authorization_code'
        )
        
        ACCESS_TOKEN = response.access_token
        print("\n[SUCCESS] Access Token generated successfully!")
        print(f"Token (Expires daily): {ACCESS_TOKEN[:10]}...")
        return ACCESS_TOKEN

    except ApiException as e:
        print(f"\n[ERROR] Failed to get Access Token:")
        print(e)
        return None

# -----------------------------------------------------------------
# 3. MAIN EXECUTION BLOCK
# -----------------------------------------------------------------
def main():
    if not API_KEY or API_KEY == "YOUR_UPSTOX_API_KEY":
        print("[CRITICAL] Please update your API_KEY and API_SECRET in config/keys.py first.")
        return

    # Step 1: Get Authorization Code
    redirected_url = generate_login_url()
    
    # Step 2: Get Access Token
    token = get_access_token(redirected_url)

    if token:
        # Savee token to a local file for the day's use
        with open("token.txt", "w") as f:
            f.write(token)
        print("\nSetup complete. Token saved to token.txt. Ready for data fetching!")

if __name__ == "__main__":
    main()