import requests

# API endpoint
url = 'http://localhost:9696/predict'
#url = "https://holy-snowflake-4008.fly.dev/predict"

# Example patient data
patient = {
  "age": 54,
  "gender": "m",
  "smoking": "yes",
  "hx_smoking": "no",
  "hx_radiothreapy": "no",
  "thyroid_function": "euthyroid",
  "physical_examination": "single_nodular_goiter-left",
  "adenopathy": "right",
  "pathology": "micropapillary",
  "focality": "multi-focal",
  "tumor": "t4a",
  "node": "n1b",
  "metastasis": "m0",
  "stage": "ii",
  "response": "structural_incomplete"
}

# Send POST request
response = requests.post(url, json=patient)

# Parse response
predictions = response.json()

print(predictions)

# Interpret prediction
if predictions['recurred']:
    print("Patient is likely to have recurrence, consider closer monitoring")
else:
    print("Patient is unlikely to have recurrence")