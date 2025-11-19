# ml-ThyRP
### A machine learning project predicting the recurrence of well differentiated thyroid cancer.
Done as the Midterm Project for the Data Talks Club Machine Learning Zoomcamp 2025.

## Problem Statement
Thyroid cancer mortality is not high. But the risk of recurrence is significant. Rescources and time can be managed much better, if we can predict the risk of recurrence in a patient. We can thus modify treatment methods accordingly and monitor patients who are at risk of recurrence more closely with regular follow-ups, while those with low risk need to come in less frequently. Automating this decision and scheduling frees up valuable time for healthcare workers and gives them a clean information set for decision making.

In this case, any patient with recurrence risk > 50% was monitored more closely, but one could also stratify the patients into specific risk categories (e.g., low, medium, high).

## The Dataset
`Thyroid_Diff.csv` in this repository. It is a dataset obtained from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/).

>> Citation: Borzooei, S., Briganti, G., Golparian, M. et al. Machine learning for risk stratification of thyroid cancer patients: a 15-year cohort study. Eur Arch Otorhinolaryngol 281, 2095–2104 (2024). https://doi.org/10.1007/s00405-023-08299-w

>> The dataset and details about it are available [here](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence).


The dataset contains 13 clinopathologic features, all categorical apart from age. The target is `recurred`.


## Approach
After initial cleaning and exploratory data analysis, three different models were trained and tuned on the data: Logistic Regression, Random Forest Classifier, and XGBoost Classifier. Their performance was compared, and XGBoost was chosen as performing the best and most consistently. This is the model that was deployed for use.

The entire training, tuning, and performance comparison can be found in `notebook.ipynb`.
Performance comparison:
![Bar Chart Model Comparison](model_performance_comparison.png)

## 🚀 How to Use This Repository

This project provides a FastAPI service for predicting thyroid cancer recurrence using an XGBoost model.
You can run it in two ways:
- In Docker, using the included Dockerfile
- Locally, using a virtual environment managed by uv

Both workflows are documented below.

You can also train the model yourself using `train.py`.

### 1. Clone this repository:
```bash
git clone https://github.com/malenaduroux/ml-ThyRP
cd mlThyRP
```
Or do this manually on Github by forking the repository. You can then create a Github Codespace to work in if you like.

### 🐳 Option A: Run Using Docker

1. Build the Docker image

```bash
docker build -t recurrence-prediction .
````

2. Run the container in interactive mode (remove it after using)

```bash
docker run -it --rm -p 9696:9696 recurrence-prediction
```

The app will not be served http://localhost:9696/predict. You can access the docs and try it out on http://localhost:9696/predict/docs, run the test file `python test.py`, or use a curl request:

```bash
curl -X POST "http://localhost:9696/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "age": 45,
           "gender": "f",
           "smoking": "no",
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
         }'
```

### 💾 Option B: Run locally using uv

1. Install uv if not installed yet
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and sync virtual environment
```bash
uv sync
```

3. Activate it
```bash
source .venv/bin/activate
```

4. Start the FastAPI app
```bash
uvicorn predict:app --host 0.0.0.0 --port 9696
```
The app is now available on http://localhost:9696/predict like with the Docker method.