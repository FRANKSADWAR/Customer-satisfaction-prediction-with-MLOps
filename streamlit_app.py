import streamlit  as st
import pandas as pd
import numpy as np

import json
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_deployment

def main():
    st.title("End to end Customer Satisfaction prediction pipeline with ZenML")
    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
    | Payment Installments   | Number of installments chosen by the customer. |  
    | Payment Value |       Total amount paid by the customer. | 
    | Price |       Price of the product. |
    | Freight Value |    Freight value of the product.  | 
    | Product Name length |    Length of the product name. |
    | Product Description length |    Length of the product description. |
    | Product photos Quantity |    Number of product published photos |
    | Product weight measured in grams |    Weight of the product measured in grams. | 
    | Product length (CMs) |    Length of the product measured in centimeters. |
    | Product height (CMs) |    Height of the product measured in centimeters. |
    | Product width (CMs) |    Width of the product measured in centimeters. |
    """
    )

    payment_sequential = st.sidebar.slider("Payment Sequential")
    payment_installments = st.sidebar.slider("Payment Installments")
    payment_value = st.sidebar.number_input("Payment Value")
    price = st.sidebar.number_input("Price")
    freight_value = st.sidebar.number_input("freight value")
    product_name_length = st.sidebar.number_input("Product name length")
    product_description_length = st.sidebar.number_input("Product Description length")
    product_photos_qty = st.sidebar.number_input("Product photos Quantity ")
    product_weight_g = st.sidebar.number_input("Product weight measured in grams")
    product_length_cm = st.sidebar.number_input("Product length (CMs)")
    product_height_cm = st.sidebar.number_input("Product height (CMs)")
    product_width_cm = st.sidebar.number_input("Product width (CMs)")

    if st.button("Predict"):
        service = prediction_service_loader(
            pipeline_name="continous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running = False
        )

        if service is None:
            st.write("No service could be found, pipeline will be run first to create a service")
            run_deployment()


        df = pd.DataFrame(
            {

                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "price": [price],
                "freight_value": [freight_value],
                "product_name_lenght": [product_name_length],
                "product_description_lenght": [product_description_length],
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weight_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm],
            }
        ) 
        json_list = json.loads(json.dumps(list(df.T.to_dict().values()))) 
        data = np.array(json_list)

        pred = service.predict(data)

        st.success(
            "Your customer satisfaction score is {}".format(pred)
        )  

    if st.button("Results"):
        st.write(
            "We have experimented with two ensemble and tree based models and compared the performance of each model. The results are as follows:"
        )

        df = pd.DataFrame(
            {
                "Models": ["LightGBM", "Xgboost"],
                "MSE": [1.804, 1.781],
                "RMSE": [1.343, 1.335],
            }
        )
        st.dataframe(df)


if __name__=="__main__":
    main()           

