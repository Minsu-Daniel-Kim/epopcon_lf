
# coding: utf-8

# In[133]:


# %load_ext autoreload
# # from ggplot import *
# %autoreload 3
from util import *

engine = get_engine()


# item = glob.glob('data/pickle/ivt_item/ivt_item_1.pkl')

def process_batch(file):
    
    batch = pd.read_pickle(file)
    
    logging.warning('stage1')
    result_lst = []
    for idx, group_by_item_id in batch.groupby('ITEM_ID'):
        tmp = list(group_by_item_id.groupby('STOCK_ID'))[0][1]    
        result_lst.append(get_feature_engineered_bundle(tmp))

    logging.warning('stage2')
    results = [result for result in result_lst if result != None]
    result_df = pd.DataFrame(results)
    logging.warning('stage3')
    # filter dataframe
    filtered_df = get_filtered_fg_df(result_df)
    
    cleaned_item_ids = filtered_df.item_id.values
    cleaned_df = batch[batch['ITEM_ID'].isin(cleaned_item_ids)]
    logging.warning('stage4')
    df_lst =[]
    print(df_lst)

    for idx, group in list(cleaned_df.groupby('ITEM_ID')):
        try:
            df_lst.append(get_sell_amount_by_item_id(group)['result'])
        except:
            continue
            
    if len(df_lst) > 0:
            
        result = pd.concat(df_lst)
        result.to_sql(con=engine, name='MWS_COLT_ITEM_SELL_AMT', if_exists='append', flavor='mysql')
        logging.warning('done with %s' % str(file))


# In[148]:


if __name__ == '__main__':
    files = glob.glob('data/pickle/ivt_item/*.pkl')[:2]
    engine = get_engine()
    add_engine_pidguard(engine)    
    Parallel(n_jobs=-1)(map(delayed(process_batch), files))

