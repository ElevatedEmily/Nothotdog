import requests


url = "http://127.0.0.1:5000/predict"
file_path = "train/nothotdog/2.jpg"


with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# Print the response from the server
print(response.status_code)  # HTTP status code
print(response.text)         # Raw response body

