import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def main():

    st.title("Some data manipulations")

    home_data = get_data()

    app_mode = st.sidebar.selectbox(
        "Choose the app mode", ["Show instructions", "Regression"]
    )  # , "Show the source code"])
    if app_mode == "Show instructions":
        st.write("To continue select a mode in the selection box to the left.")
    # elif app_mode == "Show the source code":
    #     st.code(get_file_content_as_string("./app.py"))
    elif app_mode == "Regression":
        regression(home_data)


@st.cache
def get_data():
    iowa_file_path = "./home-data-for-ml-course/train.csv"
    home_data = pd.read_csv(iowa_file_path)
    return home_data


# def get_file_content_as_string(path):
#     with open(path) as f:
#         lines = f.read()
#     return lines


def regression(home_data):

    # Create target object and call it y
    y = home_data.SalePrice

    features = [
        "LotArea",
        "YearBuilt",
        "1stFlrSF",
        "2ndFlrSF",
        "FullBath",
        "BedroomAbvGr",
        "TotRmsAbvGrd",
    ]
    home_data_extracted = home_data[["SalePrice"] + features]

    st.text(
        "This is the head of the dataframe of Iowa house prices with many covariates"
    )
    st.write(home_data_extracted.head())

    # Create X
    covariates = st.multiselect(
        "Select covariates to keep for regression:", features, features
    )
    X = home_data[covariates]

    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    dict_val_maes = {"method": [], "Val MAE": []}

    # Specify Model
    iowa_model = DecisionTreeRegressor(random_state=1)
    # Fit Model
    iowa_model.fit(train_X, train_y)
    # Make validation predictions and calculate mean absolute error
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    dict_val_maes["method"].append("DecisionTreeRegressor")
    dict_val_maes["Val MAE"].append(val_mae)

    # Using best value for max_leaf_nodes
    iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
    iowa_model.fit(train_X, train_y)
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    dict_val_maes["method"].append("DecisionTreeRegressor with max leaf nodes")
    dict_val_maes["Val MAE"].append(val_mae)

    # Define the model. Set random_state to 1
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    dict_val_maes["method"].append("RandomForestRegressor")
    dict_val_maes["Val MAE"].append(rf_val_mae)

    val_maes = pd.DataFrame(dict_val_maes).set_index("method")
    st.write(val_maes)
    st.text("(Test what happens when removing TotRmsAbvGrd)")


if __name__ == "__main__":
    main()
