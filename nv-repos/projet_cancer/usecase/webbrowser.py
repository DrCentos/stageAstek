"""import requests

url = 'https://buildnv-utkj5a52gq-ew.a.run.app'
headers = {'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjgyMjgzOGMxYzhiZjllZGNmMWY1MDUwNjYyZTU0YmNiMWFkYjViNWYiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIzMjU1NTk0MDU1OS5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImF1ZCI6IjMyNTU1OTQwNTU5LmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwic3ViIjoiMTAyMTIxODM5MzA0MTU0MTAwMzY1IiwiZW1haWwiOiJudmFsbG90QGdyb3VwZWFzdGVrLmZyIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImF0X2hhc2giOiJCN2FfZHFqTmhjVEhpSFN1UFRUZUF3IiwiaWF0IjoxNjg0MjQwMjk0LCJleHAiOjE2ODQyNDM4OTR9.FeF3PrtKPDMq4uuB0hO3AKmtnJN3Ui6esLsLhC6FMzm3xBuwnhZZbVimCSoqZVNe6DrVKwMcED5vNNEMyNWTYBa-vYEfUT5i_-IMFNug3JJNF1Gcbmlbjzly-CmwwL7jPGWHuNLEtYsXgKRD3qv84Y2W4lK-trNFkpF-ncaOIkZ8MkZNmamKmARFtj-4a0CZ8FdQ-fdg1d4UnADLHcH3YIAOAOLZySbFnvG9imvHqEr6mrDQEwIqpajS_vyEwgq-QjnYDEpDb092aesF2luPEUqEON4aINvWQNoupqC3At4-lKUqEKKR2lNKFvPGcjcJ_mFGE1DPzvCMTGMYan860A'}
response = requests.get(url, headers=headers)

print(response.text)
"""
import os

url = 'https://buildnv-utkj5a52gq-ew.a.run.app'
headers = {'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjgyMjgzOGMxYzhiZjllZGNmMWY1MDUwNjYyZTU0YmNiMWFkYjViNWYiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIzMjU1NTk0MDU1OS5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImF1ZCI6IjMyNTU1OTQwNTU5LmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwic3ViIjoiMTAyMTIxODM5MzA0MTU0MTAwMzY1IiwiZW1haWwiOiJudmFsbG90QGdyb3VwZWFzdGVrLmZyIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImF0X2hhc2giOiJCN2FfZHFqTmhjVEhpSFN1UFRUZUF3IiwiaWF0IjoxNjg0MjQwMjk0LCJleHAiOjE2ODQyNDM4OTR9.FeF3PrtKPDMq4uuB0hO3AKmtnJN3Ui6esLsLhC6FMzm3xBuwnhZZbVimCSoqZVNe6DrVKwMcED5vNNEMyNWTYBa-vYEfUT5i_-IMFNug3JJNF1Gcbmlbjzly-CmwwL7jPGWHuNLEtYsXgKRD3qv84Y2W4lK-trNFkpF-ncaOIkZ8MkZNmamKmARFtj-4a0CZ8FdQ-fdg1d4UnADLHcH3YIAOAOLZySbFnvG9imvHqEr6mrDQEwIqpajS_vyEwgq-QjnYDEpDb092aesF2luPEUqEON4aINvWQNoupqC3At4-lKUqEKKR2lNKFvPGcjcJ_mFGE1DPzvCMTGMYan860A'}
chrome_path = 'C:\Program Files\Google\Chrome\Application/chrome.exe %s'
os.system(f'start chrome {url} --new-window --user-data-dir=\\temp --disable-extensions --disable-plugins --disable-popup-blocking --disable-translate --no-first-run --no-default-browser-check --disable-web-security --user-data-dir=\\temp --disable-site-isolation-trials --disable-site-isolation-for-policy --headers="{headers}"')



