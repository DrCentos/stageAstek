
import google.auth
import google.auth.transport.requests
import requests
import webbrowser

credentials, _ = google.auth.default()
credentials.refresh(google.auth.transport.requests.Request())
identity_token = credentials.id_token

auth_header = {"Authorization": "Bearer " + identity_token}
#response = requests.get("${SERVICE_URL}/partners", headers=auth_header)
response = requests.get("https://nv-utkj5a52gq-ew.a.run.app", headers=auth_header)

redirect_url = response.url
partners = response
webbrowser.open(redirect_url)
print(partners)

