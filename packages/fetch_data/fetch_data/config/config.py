import pathlib
import os
import fetch_data
import pandas as pd

PACKAGE_ROOT = pathlib.Path(fetch_data.__file__).resolve().parent

DATA_RAW_DIR = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(PACKAGE_ROOT))),'data'),'raw')
DATA_RAW_PROCESS = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(PACKAGE_ROOT))),'data'),'preprocess')

# data
df_customers =  'olist_customers_dataset.csv'
df_order_items =  'olist_order_items_dataset.csv'
df_order_payments =  'olist_order_payments_dataset.csv'
df_order_reviews =  'olist_order_reviews_dataset.csv'
df_orders =  'olist_orders_dataset.csv'
df_order_products =  'olist_products_dataset.csv'
df_order_sellers =  'olist_sellers_dataset.csv'