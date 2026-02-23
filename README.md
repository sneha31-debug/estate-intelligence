# 🏠 AIML Estate Intelligence

## Problem Statement
The objective of this project is to build a robust Machine Learning model to predict house prices based on various features such as area, number of bedrooms, bathrooms, stories, and other amenities. Accurate price prediction helps real estate professionals and potential buyers make informed decisions in a dynamic housing market.

**Dataset Link:** The model is trained on the [Housing Prices Dataset](https://www.kaggle.com/datasets/ashydv/housing-dataset) (or internal CSV provided).

---

## Architecture Diagram (Workflow)

```mermaid
graph TD
    A[Raw Data (Housing.csv)] -->|Data Preprocessing| B(Cleaned Data)
    B -->|Outlier Treatment & Scaling| C{Feature Engineering}
    C -->|Split Data| D[Training Set]
    C -->|Split Data| E[Testing Set]
    D --> F(Model Training)
    F -->|Linear Regression| G{Evaluate Models}
    F -->|Decision Tree| G
    F -->|Random Forest| G
    E --> G
    G -->|Select Best Model| H[Saved Model .pkl]
    H --> I[Streamlit Interface app.py]
    user([User Input]) --> I
    I -->|Predict Price| user
```

---

## Input / Output Specifications

### Model Inputs
The prediction model accepts the following features as inputs:
- **Numerical Inputs:** 
  - `area` (int): Total area of the house in square feet.
  - `bedrooms` (int): Number of bedrooms.
  - `bathrooms` (int): Number of bathrooms.
  - `stories` (int): Number of stories/floors.
  - `parking` (int): Capacity of parking available.
- **Categorical Inputs (Encoded to 1/0 or ordinal values):**
  - `mainroad`: Access to main road (Yes/No).
  - `guestroom`: Presence of a guest room (Yes/No).
  - `basement`: Presence of a basement (Yes/No).
  - `hotwaterheating`: Availability of hot water heating (Yes/No).
  - `airconditioning`: Availability of AC (Yes/No).
  - `prefarea`: Located in a preferred neighborhood (Yes/No).
  - `furnishingstatus`: Status of furnishing (Unfurnished=0, Semi-furnished=1, Furnished=2).

### Model Output
- **Prediction:** Continuous numerical value (`price`) predicting the estimated price of the house in INR (₹).

---

## Model Performance

*(This section will be updated by Krish once Model Training and Evaluation is complete)*

- **Baseline Model (Linear Regression):**
  - RMSE: TBD
  - R² Score: TBD
- **Tree-Based Models:**
  - Decision Tree R²: TBD
  - Random Forest R²: TBD

-- Best Model: **[TBD]**

---

## Installation & Local Setup

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd estate-intelligence
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## Deployment
This application is designed to be easily deployed on **Streamlit Cloud** or **HuggingFace Spaces**. Currently, a placeholder UI exists waiting for final model integration.

**Live Link:** [Pending Deployment URL]