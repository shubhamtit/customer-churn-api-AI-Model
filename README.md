# Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to **churn or stay** using Logistic Regression.

This project covers the complete journey from preparing the data and training the model to building a FastAPI API and deploying it with Docker.

## 🚀 Project Workflow

```text
Dataset
   ↓
EDA & Data Preprocessing
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Scikit-learn Pipeline
   ↓
Joblib
   ↓
FastAPI
   ↓
Docker
   ↓
Docker Hub
```

## 🧠 What I Learned

* Exploratory Data Analysis (EDA)
* Feature selection
* Train/Test splitting
* Feature scaling using `StandardScaler`
* Logistic Regression for binary classification
* Confusion Matrix
* Accuracy, Precision, Recall and F1 Score
* Scikit-learn Pipelines
* Saving models using Joblib
* Building REST APIs with FastAPI
* Dockerizing an ML application
* Publishing Docker images to Docker Hub

## 📊 Dataset

The dataset contains synthetic customer information with the following features:

| Feature           | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `age`             | Customer age                                            |
| `tenure_months`   | Number of months the customer has been with the company |
| `monthly_charges` | Customer's monthly charges                              |
| `support_calls`   | Number of customer support calls                        |
| `contract_months` | Contract duration                                       |
| `churn`           | Target variable: `0` = No Churn, `1` = Churn            |

> The dataset was created for learning purposes and is synthetic.

## 🤖 Machine Learning Model

I used **Logistic Regression** because this is a binary classification problem.

The model predicts:

```text
0 → No Churn
1 → Churn
```

The model also provides a probability of churn.

For example:

```text
Prediction: Churn
Churn Probability: 0.88
```

## 📈 Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Example results from the test set:

```text
Accuracy : 0.805
Precision: 0.832
Recall   : 0.825
F1 Score : 0.828
```

These results are based on the synthetic dataset and the particular train/test split used in this project.

## 🔗 ML Pipeline

Instead of separately handling scaling and prediction, I combined them into a Scikit-learn Pipeline:

```text
Input Data
    ↓
StandardScaler
    ↓
Logistic Regression
    ↓
Prediction
```

The complete pipeline was saved using Joblib:

```text
customer_churn_pipeline.pkl
```

This allows the API to receive raw customer data and automatically perform the required preprocessing before making a prediction.

## 🌐 FastAPI

A FastAPI application was created to serve the trained model.

### Endpoint

```text
POST /prediction
```

### Example Request

```json
{
  "age": 25,
  "tenure_months": 4,
  "monthly_charges": 2500,
  "support_calls": 7,
  "contract_months": 1
}
```

### Example Response

```json
{
  "prediction": "Churn",
  "churn_probability": 0.88,
  "message": "Customer is predicted to churn."
}
```

FastAPI's Swagger UI can be used to test the API:

```text
/docs
```

## 🐳 Docker

The FastAPI application was containerized using Docker.

### Build the image

```bash
docker build -t customer-churn-api .
```

### Run the container

```bash
docker run -p 8000:8000 customer-churn-api
```

The API will then be available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

## ☁️ Docker Hub

The Docker image has been published to Docker Hub.

**Docker Hub:** https://hub.docker.com/repository/docker/shubhamsinghtit/customer-churn-api/general

You can pull and run the image using:

```bash
docker push shubhamsinghtit/customer-churn-api:tagname
```

```bash
docker run -p 8000:8000 shubhamsinghtit/customer-churn-api:tagname
```

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* FastAPI
* Uvicorn
* Docker
* Docker Hub

## 📁 Project Structure

```text
Project2/
│
├── main.py
├── customer_churn.csv
├── customer_churn_pipeline.pkl
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## 🎯 Future Learning

Next, I plan to explore:

* MLOps
* CI/CD
* Cloud deployment
* Model monitoring
* Model versioning
* Production ML systems

---

### 👨‍💻 Learning Project

This project was built as part of my Machine Learning learning journey, with the goal of understanding not only how to train a model, but also how to turn it into a deployable application.
