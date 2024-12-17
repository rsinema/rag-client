import requests

response = requests.get("http://localhost:8000/retrieve", params={"query": "Does Riley know how to use C#?"})
for result in response.json():
    print(result)