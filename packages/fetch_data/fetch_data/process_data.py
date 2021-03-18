import pandas as pd
import os
from fetch_data.config import config

df_customers = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_customers))
df_order_items = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_order_items))
df_order_payments = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_order_payments))
df_order_reviews = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_order_reviews))
df_orders = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_orders))
df_order_products = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_order_products))
df_order_sellers = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_order_sellers))

df = pd.merge(df_orders,df_order_payments, on="order_id", how = 'inner')
df = df.merge(df_order_reviews, on="order_id", how = 'inner')
df = df.merge(df_customers, on="customer_id", how = 'inner')
df = df.merge(df_order_items, on="order_id", how = 'inner')
df = df.merge(df_order_sellers, on="seller_id", how = 'inner')
df = df.merge(df_order_products, on="product_id", how = 'inner')

df.to_csv(os.path.join(config.DATA_RAW_PROCESS, 'data.csv'), index = False)