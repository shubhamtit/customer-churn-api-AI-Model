import numpy as np
import pandas as pd

# Make results reproducible
np.random.seed(42)

number_of_customers = 1000


# -----------------------------
# Generate customer information
# -----------------------------

age = np.random.randint(18, 70, number_of_customers)

tenure_months = np.random.randint(1, 73, number_of_customers)

monthly_charges = np.random.randint(
    500,
    3001,
    number_of_customers
)

support_calls = np.random.randint(
    0,
    11,
    number_of_customers
)

contract_months = np.random.choice(
    [1, 6, 12, 24],
    number_of_customers
)


# -----------------------------
# Generate churn
# -----------------------------

churn_score = (
    -0.04 * tenure_months
    + 0.35 * support_calls
    + 0.0003 * monthly_charges
    - 0.03 * contract_months
    + np.random.normal(0, 1, number_of_customers)
)

churn_probability = 1 / (1 + np.exp(-churn_score))

churn = (churn_probability >= 0.5).astype(int)


# -----------------------------
# Create DataFrame
# -----------------------------

data = pd.DataFrame({
    "age": age,
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "support_calls": support_calls,
    "contract_months": contract_months,
    "churn": churn
})


# -----------------------------
# Save dataset
# -----------------------------

data.to_csv(
    "customer_churn.csv",
    index=False
)

print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nChurn distribution:")
print(data["churn"].value_counts())
