import requests

url = "http://127.0.0.1:5000/predict"
total_cases = int(input("Enter the number of reported cases: "))
data = {"Total_Cases": total_cases}
response = requests.post(url, json=data)
response.raise_for_status()
print(response.json())