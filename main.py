from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# import tensorflow as tf
from PIL import Image
import pytesseract
from pydantic import BaseModel

data = {
    "Date": pd.date_range(start="2021-01-01", periods=36, freq="M"),
    "Food": np.random.randint(5000, 15000, 36),
    "Electricity": np.random.randint(1000, 5000, 36),
    "Transportation": np.random.randint(2000, 8000, 36),
    "Others": np.random.randint(1000, 5000, 36),
    "Rent/EMI": np.random.randint(10000, 30000, 36),
    "Insurance": np.random.randint(2000, 8000, 36),
    "Paid services/subscription": np.random.randint(500, 2000, 36),
}

df = pd.DataFrame(data)
# Step 2: Extract month and year from the date
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# Drop the Date column
df.drop(columns=["Date"], inplace=True)

# Step 3: Predict monthly budget allocation for each category
category_columns = [
    "Food",
    "Electricity",
    "Transportation",
    "Others",
    "Rent/EMI",
    "Insurance",
    "Paid services/subscription",
]

X = df[["Year", "Month"]]
y = df[category_columns]
# Train a regression model for each category
models = {}
for category in category_columns:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y[category], test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    models[category] = model

# Predict budgets for the next month
next_month = pd.DataFrame({"Year": [2024], "Month": [6]})
predicted_budgets = {}
for category, model in models.items():
    predicted_budgets[category] = model.predict(next_month)[0]

# Calculate the total predicted budget
total_predicted_budget = sum(predicted_budgets.values())


# Function to normalize predicted budgets based on user's income
def normalize_budgets(predicted_budgets, user_income, total_predicted_budget):
    normalized_budgets = {}
    for category, budget in predicted_budgets.items():
        normalized_budgets[category] = (budget / total_predicted_budget) * user_income
    return normalized_budgets


# Step 4: Analyze how changes in one category affect others
correlation_matrix = df[category_columns].corr()


# Function to recommend budget adjustments
def recommend_budget_adjustments(category, change_amount, normalized_budgets):
    adjustments = {}
    correlations = correlation_matrix[category]
    sorted_correlations = correlations.abs().sort_values(ascending=False)
    for other_category in sorted_correlations.index[1:]:
        adjustment = change_amount * correlations[other_category]
        adjustments[other_category] = adjustment
    return adjustments


# Step 5: Recommend the best payment mode based on offers
offers = {
    "Food": {"cash": 0, "credit_card": 5, "debit_card": 2},
    "Electricity": {"cash": 0, "credit_card": 3, "debit_card": 1},
    "Transportation": {"cash": 0, "credit_card": 4, "debit_card": 3},
    "Others": {"cash": 0, "credit_card": 2, "debit_card": 1},
    "Rent/EMI": {"cash": 0, "credit_card": 2, "debit_card": 1},
    "Insurance": {"cash": 0, "credit_card": 5, "debit_card": 2},
    "Paid services/subscription": {"cash": 0, "credit_card": 4, "debit_card": 3},
}


def recommend_payment_mode(category, available_modes):
    category_offers = {
        mode: cashback
        for mode, cashback in offers[category].items()
        if available_modes[mode]
    }
    return max(category_offers, key=category_offers.get)


# Step 6: User Input and Output


class llm(BaseModel):
    prompt: str
    stream: bool = False
    model: str = "llama3"


class stats(BaseModel):
    transaction_type: str
    amt: float
    catagory: str = None
    priority: str


class initial_stats(BaseModel):
    income: float
    savings: float


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this according to your needs. "" allows all origins.
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE",
    ],  # Adjust the allowed methods as needed.
    allow_headers=["*"],  # Adjust the allowed headers as needed.
)


@app.get("/")
def home():
    return "api is up"


class predictReturn(BaseModel):
    Food: float
    Electricity: float
    Transportation: float
    Others: float
    Rent_EMI: float
    Insurance: float
    Paid_services_subscription: float


@app.post("/recomended", response_model=predictReturn)
def recomend(initial_stats: initial_stats):
    stats = dict(initial_stats)
    monthly_income = stats["income"]
    savings_amount = stats["savings"]
    remaining_income = monthly_income - savings_amount

    # Normalize the predicted budget according to the user's income
    normalized_budgets = normalize_budgets(
        predicted_budgets, remaining_income, total_predicted_budget
    )
    recomended = {}
    for category, budget in normalized_budgets.items():
        if category == "Paid services/subscription":
            recomended["Paid_services_subscription"] = float(f"{budget:.2f}")
        elif category == "Rent/EMI":
            recomended["Rent_EMI"] = float(f"{budget:.2f}")
        else:
            recomended[f"{category}"] = float(f"{budget:.2f}")
    return recomended


class dateType(BaseModel):
    day: int
    month: int
    year: int


class dataType(BaseModel):
    category: str
    amount: float
    description: str


class orcReturnRype(BaseModel):
    date: dateType | None
    data: list[dataType]


class Form(BaseModel):
    file: str


import base64
from io import BytesIO


@app.post("/ocr", response_model=orcReturnRype)
def ocr(formData: Form):
    try:
        file = formData.file
        file = file.split(",")[-1]
        file = base64.b64decode(file)

        # Base64 to image
        image = Image.open(BytesIO(file))

        # Perform OCR
        text = pytesseract.image_to_string(image)

        data = text.split("\n")
        data = [i for i in data if i]
        data = [i for i in data if "INVOICE" not in i]

        if not data:
            return {"date": None, "data": []}

        date = data[0]
        data = data[1:]

        finalData = []
        for i in range(0, len(data), 2):
            try:
                category, amount = data[i].rsplit(" ", 1)
                description = data[i + 1]
                finalData.append(
                    {
                        "category": category,
                        "amount": float(amount),
                        "description": description,
                    }
                )
            except Exception as e:
                print(f"Error processing line {data[i]}: {e}")
                return {"date": None, "data": []}

        date = date.split(":")[-1].strip().split("-")
        day = date[0]
        month = date[1]
        year = date[2]

        return {
            "date": {
                "day": int(day),
                "month": int(month),
                "year": int(year),
            },
            "data": finalData,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"date": None, "data": []}


@app.post("/chat")
def get_llm(llm: llm):
    data = dict(llm)
    headers = {"Content-Type": "application/json"}
    resp = requests.post(
        "http://localhost:11434/api/generate", headers=headers, data=json.dumps(data)
    ).json()

    return {"text": resp["response"]}
