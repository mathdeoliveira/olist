from kaggle.api.kaggle_api_extended import KaggleApi
from fetch_data.config import config
import zipfile
import os
from os.path import isfile, join
import time
import pandas as pd


# autenticando na api do kaggle, necessario criar o arquivo .json
api = KaggleApi()
api.authenticate()


# download dos dados do primeiro dataset
for file in config.data_1:
    api.dataset_download_file('olistbr/brazilian-ecommerce',
                              file_name=file,
                              path=config.DATA_RAW_DIR)


# download dos dados do segundo dataset
for file in config.data_2:
    api.dataset_download_file('olistbr/marketing-funnel-olist',
                              file_name=file,
                              path=config.DATA_RAW_DIR)


# capturando os nomes dos arquivos que fizeram download zipados
files_names = [f for f in os.listdir(config.DATA_RAW_DIR) if isfile(join(config.DATA_RAW_DIR, f)) if f.endswith('.zip')]

time.sleep(5)

# unzip dos arquivos e deletando os arquivos zipados
for file in files_names:
        with zipfile.ZipFile(os.path.join(config.DATA_RAW_DIR, file), 'r') as zipref:
            zipref.extractall(config.DATA_RAW_DIR)


for file in files_names:
    os.remove(os.path.join(config.DATA_RAW_DIR, file))


df_customers = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_customers))
df_order_items = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_order_items))
df_order_payments = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_order_payments))
df_order_reviews = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_order_reviews))
df_orders = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_orders))
df_order_products = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_order_products))
df_sellers = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_sellers))
df_closed_deals = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_closed_deals))
df_marketing_qualified_leads = pd.read_csv(os.path.join(config.DATA_RAW_DIR, config.df_marketing_qualified_leads))

df = pd.merge(df_orders, df_order_payments, on="order_id", how = 'inner')
df = df.merge(df_order_reviews, on="order_id", how='inner')
df = df.merge(df_customers, on="customer_id", how='inner')
df = df.merge(df_order_items, on="order_id", how='inner')
df = df.merge(df_sellers, on="seller_id", how='inner')
df = df.merge(df_order_products, on="product_id", how='inner')

mf = df_marketing_qualified_leads.merge(df_closed_deals, on='mql_id', how='left')
mf_sellers = mf.merge(df_sellers, on='seller_id', how='left')
mf_items = mf.merge(df_order_items, on='seller_id', how='left')

df.to_csv(os.path.join(config.DATA_PROCESSED, 'data_ecommerce.csv'), index=False)
mf_items.to_csv(os.path.join(config.DATA_PROCESSED, 'data_mkt_funnel.csv'), index=False)

for file in config.data_1+config.data_2:
    os.remove(os.path.join(config.DATA_RAW_DIR, file))

