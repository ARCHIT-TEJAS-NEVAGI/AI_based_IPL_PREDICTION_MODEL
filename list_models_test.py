# list_models_test.py
import os, json, sys
from google import genai

KEY = os.getenv("GEMINI_API_KEY") or ""
if not KEY:
    print("ERROR: No GEMINI_API_KEY in env. If you use .streamlit/secrets.toml, set GEMINI_API_KEY there or export it before running.")
    sys.exit(1)

try:
    client = genai.Client(api_key=KEY)
    print("Client created OK")
except Exception as e:
    print("Client creation FAILED:", e)
    sys.exit(1)

# Try list_models via client (if available)
try:
    if hasattr(client, "list_models"):
        print("Calling client.list_models() ...")
        models_resp = client.list_models()
        try:
            # print repr and to_dict if possible
            if hasattr(models_resp, "to_dict"):
                md = models_resp.to_dict()
                print("models (to_dict):")
                print(json.dumps(md, indent=2))
            else:
                print("models (raw):")
                print(models_resp)
        except Exception as e:
            print("Could not to_dict() models_resp:", e)
            print(models_resp)
    else:
        print("client.list_models not available on this SDK client.")
except Exception as e:
    print("list_models call FAILED:", repr(e))

# Try top-level genai.list_models if the SDK exposes it
try:
    import google
    if hasattr(google, "genai") and hasattr(google.genai, "list_models"):
        print("Calling google.genai.list_models() ...")
        resp = google.genai.list_models()
        try:
            if hasattr(resp, "to_dict"):
                print(json.dumps(resp.to_dict(), indent=2))
            else:
                print(resp)
        except Exception as e:
            print("Could not to_dict resp:", e)
            print(resp)
    else:
        print("google.genai.list_models not available.")
except Exception as e:
    print("Top-level genai.list_models call FAILED:", repr(e))
