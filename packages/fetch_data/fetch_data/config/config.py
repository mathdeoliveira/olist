import pathlib
import os
import fetch_data
import pandas as pd

PACKAGE_ROOT = pathlib.Path(fetch_data.__file__).resolve().parent

DATA_RAW_DIR = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(PACKAGE_ROOT))),'data'),'raw')

DATA_PROCESSED = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(PACKAGE_ROOT))),'data'),'preprocess')

# data
data_1 = ['olist_customers_dataset.csv',
        'olist_order_items_dataset.csv',
        'olist_order_payments_dataset.csv',
        'olist_order_reviews_dataset.csv',
        'olist_orders_dataset.csv',
        'olist_products_dataset.csv',
        'olist_sellers_dataset.csv']

data_2 = ['olist_closed_deals_dataset.csv',
          'olist_marketing_qualified_leads_dataset.csv']

df_customers =  'olist_customers_dataset.csv'
df_order_items =  'olist_order_items_dataset.csv'
df_order_payments =  'olist_order_payments_dataset.csv'
df_order_reviews =  'olist_order_reviews_dataset.csv'
df_orders =  'olist_orders_dataset.csv'
df_order_products =  'olist_products_dataset.csv'
df_sellers =  'olist_sellers_dataset.csv'
df_closed_deals = 'olist_closed_deals_dataset.csv'
df_marketing_qualified_leads = 'olist_marketing_qualified_leads_dataset.csv'