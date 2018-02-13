
# coding: utf-8

# In[2]:


# %load_ext autoreload
# # from ggplot import *
# %autoreload 3
from util import *


# In[378]:


# item = glob.glob('data/pickle/ivt_item/ivt_item_1.pkl')

def process_batch(file):
    
    batch = pd.read_pickle(file)
    
    result_lst = []
    for idx, group_by_item_id in batch.groupby('ITEM_ID'):
        tmp = list(group_by_item_id.groupby('STOCK_ID'))[0][1]    
        result_lst.append(get_feature_engineered_bundle(tmp))


    results = [result for result in result_lst if result != None]
    result_df = pd.DataFrame(results)
    
    # save feature engineered df
    result_df.to_pickle('data/pickle/ivt_item_feature_engineered/%s' % str(file.split('/')[-1]))
    
    # filter dataframe
    filtered_df = get_filtered_fg_df(result_df)

    
    cleaned_item_ids = filtered_df.item_id.values
    cleaned_df = batch[batch['ITEM_ID'].isin(cleaned_item_ids)]
    df_lst =[]
    
    # save images
    save_img(cleaned_df)


    for idx, group in list(cleaned_df.groupby('ITEM_ID')):
        try:
            df_lst.append(get_sell_amount_by_item_id(group))
        except:
            continue
            
    if len(df_lst) > 0:
            
        result = pd.concat(df_lst)
        result.to_sql(con=engine, name='MWS_COLT_ITEM_SELL_AMT_DEV', if_exists='append', flavor='mysql')
        logging.warning('done with %s' % str(file))


# In[379]:


def map_clean_up_target_df(stock_id, group_df):

    tmp_df = clean_up_target_df(group_df)[['sell_impute', 'STOCK_AMOUNT', 'STOCK_AMOUNT_imputed']]
    tmp_df['STOCK_ID'] = stock_id
    tmp_df.columns = ['SELL_AMOUNT', 'STOCK_AMOUNT', 'STOCK_AMOUNT_imputed', 'STOCK_ID']

    return tmp_df


# In[380]:


def get_sell_amount_by_item_id(df, add_sell_amount=False):
    collect_day = df.COLLECT_DAY.values[0]
    reg_id = df.REG_ID.values[0]
    
    tmp_lst = []
    for stock_id, group_df in list(group.groupby('STOCK_ID')):
        tmp_lst.append(map_clean_up_target_df(stock_id, group_df))    
    result = pd.concat(tmp_lst)
    
    
#     df_pivot = df.pivot_table(index='REG_DT', columns='STOCK_ID', values='STOCK_AMOUNT')
#     sell_amount_by_stock = df_pivot.apply(map_clean_up_target_df)

#     if add_sell_amount:
#         sell_amount_total = sell_amount_by_stock.sum(axis=1)
#         result = pd.DataFrame(sell_amount_total)
#         result.columns = ['SELL_AMOUNT']
#         result['REG_ID'] = reg_id
#     else:
#         sell_amount_by_stock['REG_DT'] = sell_amount_by_stock.index
#         result = pd.melt(sell_amount_by_stock, id_vars=["REG_DT"], var_name="STOCK_ID", value_name="SELL_AMOUNT")

    item_id = df.ITEM_ID.values[0]
    result['ITEM_ID'] = item_id
    result['REG_ID'] = reg_id
    result['UPT_DT'] = pd.to_datetime('now')
    result['COLLECT_DAY'] = collect_day
    result['UPT_ID'] = 'FILTER ALGO'

    return result


# In[383]:


if __name__ == '__main__':
    files = glob.glob('data/pickle/ivt_item/*.pkl')[:2]
    engine = get_engine()
    add_engine_pidguard(engine)    
    Parallel(n_jobs=-1)(map(delayed(process_batch), files))

