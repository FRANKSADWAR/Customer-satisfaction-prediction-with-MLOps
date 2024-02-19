# MLOPs-Concepts

## What is MLOps
Machine learning operations is a set of practices to design, deploy and maintain 
machine learning in production continously, reliably and effeciently.

- The focus is on machine learning that is used in a business use case to bring value to the 
business. This means that the machine learning will not be 'living' in the local 
notebooks but rather being used by other services.

- MLOPs brings aims to structure the machine learning project to realize value for
the business.

## Benefits of MLOps
- Speed in deployment and development
- Reliability and security
- Automation

# Use Case: Customer Reviews Score prediction

## The Data
This is Brazil's Olist e-commerce public dataaset made at Olist Store. It has information of about 100k orders from 2016 to 2018 made at multipl marketplaces in Brazil.

The dataset allows viewing and combining multiple dimensions such as order status, price, payment and freight to customer locations, product attributes and finally reviews written by customers.

## Area of interest: 
The area of interest is the review score. Can the company be able to know the reviews provided by the customers based on the way the products were
delivered to them ?
This will help the company adjust their business modes, i.e providing orders in due time. This in turn leads to customer retention. 




- Create the virtual environment for this project

```
pip install -r requirements.txt
```

```
conda create -n customer_satisfaction python=3.9.10 anaconda
```



```
zenml experiment-tracker register mlflow_tracker_customer --flavor=mlflow 
```

```
 zenml model-deployer register mlflow_customer --flavor=mlflow
```

```
zenml stack register mlflow_stack -a default -o default -d mlflow_customer -e mlflow_tracker_customer --set
```