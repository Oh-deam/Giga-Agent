import requests

url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

payload={
  'scope': 'GIGACHAT_API_PERS'
}
headers = {
  'Content-Type': 'application/x-www-form-urlencoded',
  'Accept': 'application/json',
  'RqUID': '1ab7b42c-e673-410f-98fd-6c34870a4033',
  'Authorization': 'Basic <Authorization key>'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)