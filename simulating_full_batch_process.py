
# coding: utf-8

# In[8]:


from util import *
from time import gmtime, strftime
from pytz import timezone
from datetime import datetime
from sqlalchemy import ForeignKey, Table, Column, String, Integer, Float, Boolean, MetaData, select
from joblib import Parallel, delayed


# In[13]:


epopcon_db = Epopcon_db()

wspider_engine = epopcon_db.get_engine(production=True)
wspider_temp_engine = epopcon_db.get_engine(production=False)


# In[10]:


def impute_data(target):
    
    # cluster inventory data points
    n_cluster, label = get_label_from_dbscan(target, eps=0.15, min_samples=3)
    target['label'] = label
    target = target[['STOCK_AMOUNT', 'label', 'REG_DT']]
    labels = target.label.unique()
    
    # resample to a daily scale
    target = target.set_index('REG_DT')
    target = target.resample('1D').first()
    
    # placeholding
    target['STOCK_AMOUNT_imputed'] = target['STOCK_AMOUNT']

    # interpolate data points based on cluster group
    for label in labels:
        idx = np.where(target.label.values == label)[0]
        if len(idx) == 0:
            continue
        start_v = min(idx)
        end_v = max(idx)
        target.loc[start_v:end_v+1, 'STOCK_AMOUNT_imputed'] = target['STOCK_AMOUNT'][start_v:end_v+1].interpolate(method='from_derivatives')

    # interpolate data points based on global data points
    target['STOCK_AMOUNT_imputed'] = target['STOCK_AMOUNT'].interpolate(method='from_derivatives')
    
    # round STOCK_AMOUNT_imputed to make it cleaner
    target['STOCK_AMOUNT_imputed'] = target.STOCK_AMOUNT_imputed.round()

    # calculate sell amount 
    target['sell'] = np.append([0], np.negative(np.diff(target.STOCK_AMOUNT_imputed)))
    target.loc[target['sell'].values < 0, 'sell'] = np.nan
    target.sell.astype(float)
    
    # calculate z-score for thresholding
    target['zscore'] = np.abs(target.sell - target.sell.mean() / max(0.0001, target.sell.std()))

    # get rid of outliers 
    target.loc[target['zscore'] > 4, 'sell'] = np.nan
    
    # prepare matrix for data imputation using KNN based on dayofweek
    target['weekday_name'] = target.index.dayofweek
    X_incomplete = target[['sell', 'weekday_name']].values

    # run KNN to calculate sell_impute (imputed version of sell amount)
    try:
        X_filled_knn = KNN(k=1).complete(X_incomplete)
        target['sell_impute'] = X_filled_knn[:,0]
    except:
        target['sell_impute'] = target['sell']
    
    # placeholding
    target['STOCK_AMOUNT_imputed_trimed'] = target['STOCK_AMOUNT_imputed']
    
    # get rid of jumpbs
    cond = np.append([0], np.negative(np.diff(target.STOCK_AMOUNT_imputed))) < 0
    target.loc[cond, 'STOCK_AMOUNT_imputed_trimed'] = np.nan

    return target

# TODO optimize parameters using ML

def get_filtered_fg_df(feature_engineered_df):
    static_item_ids = feature_engineered_df.item_id[(feature_engineered_df.std_in_cluster == 0.0)].values
    data_df_cleaned = feature_engineered_df[feature_engineered_df.mean_in_cluster.notnull()]
    purified_df = data_df_cleaned[(data_df_cleaned.ratio_drop < 0.3)
#                           & (data_df_cleaned.ratio_same_value < 0.3)
                          & (data_df_cleaned.n_jumps <= 3)
                          & (data_df_cleaned.n_days >= 3)
#                           & (data_df_cleaned.std_in_cluster > 0.2)
                          & (data_df_cleaned.std_in_cluster < 4)
                          & (data_df_cleaned.ratio_of_na < 0.5)
#                           & (data_df_cleaned.n_unique_stock_id < 50)
                                 ]
    return purified_df, static_item_ids

def get_sell_amount_by_item_id(df, add_sell_amount=False):
    
    collect_day = df.COLLECT_DAY.values[0]
    reg_id = df.REG_ID.values[0]
    
    imputed_df_lst = []
    for stock_id, group_df in list(df.groupby('STOCK_ID')):
        
        imputed_df = impute_data(group_df)[['sell_impute', 'STOCK_AMOUNT', 'STOCK_AMOUNT_imputed_trimed']]
        imputed_df['STOCK_ID'] = stock_id        
        imputed_df_lst.append(imputed_df)
        
    imputed_df = pd.concat(imputed_df_lst)
    imputed_df.columns = ['SELL_AMOUNT', 'STOCK_AMOUNT', 'REVISE_STOCK_AMOUNT', 'STOCK_ID']
    imputed_df['ITEM_ID'] = df.ITEM_ID.values[0]
    imputed_df['REG_ID'] = reg_id
    imputed_df['UPT_DT'] = pd.to_datetime(datetime.now(timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S"))
    imputed_df['COLLECT_DAY'] = collect_day
    imputed_df['UPT_ID'] = 'FILTER ALGO'

    return imputed_df

def insert_extracted_feature(extracted_feature_df):
    extracted_feature_df = extracted_feature_df.where((pd.notnull(extracted_feature_df)), None)
    query = """INSERT IGNORE INTO MWS_COLT_ITEM_EXTRACTED_FEATURE %s VALUES %s """ % (tuple(extracted_feature_df.columns), tuple(['%s' for _ in range(len(extracted_feature_df.columns))]))
    query = query.replace("'", "")
    wspider_temp_engine.execute(query, [tuple(x) for x in extracted_feature_df.values])

def insert_sell_amt(sell_amt_df):
    sell_amt_df = sell_amt_df.where((pd.notnull(sell_amt_df)), None)
    query = """INSERT IGNORE INTO MWS_COLT_ITEM_SELL_AMT_DEV %s VALUES %s """ % (tuple(sell_amt_df.columns), tuple(['%s' for _ in range(len(sell_amt_df.columns))]))
    query = query.replace("'", "")
    wspider_temp_engine.execute(query, [tuple(x) for x in sell_amt_df.values])
    
    query2 = """INSERT IGNORE INTO MWS_COLT_ITEM_SELL_AMT %s VALUES %s """ % (tuple(sell_amt_df.columns), tuple(['%s' for _ in range(len(sell_amt_df.columns))]))
    query2 = query2.replace("'", "")
    wspider_engine.execute(query2, [tuple(x) for x in sell_amt_df.values])


# In[11]:


def process_full_batch(batches, save_db=True, save_img=False, save_fe=True):
    
    # select multiple items
    idx, query = batches
    batch = pd.read_sql_query("SELECT * FROM MWS_COLT_ITEM_IVT WHERE ITEM_ID in %s" % query, wspider_engine)
    
    # extract features by stock id
    result_lst = []
    for idx, group_by_item_id in batch.groupby('ITEM_ID'):
        tmp = list(group_by_item_id.groupby('STOCK_ID'))[0][1]    
        result_lst.append(get_feature_engineered_bundle(tmp))

    # clean up extracted feature df
    extracted_feature_df = pd.DataFrame([result for result in result_lst if result != None])
            
    
    try:
        # filter dataframe based on extraction criteria
        filtered_df, static_item_ids = get_filtered_fg_df(extracted_feature_df)
        
        # filtered df
        cleaned_item_ids = filtered_df.item_id.values
        cleaned_df = batch[batch['ITEM_ID'].isin(cleaned_item_ids)]
        
        # label extracted feature df
        extracted_feature_df['condition_clean'] = 0
        extracted_feature_df.loc[extracted_feature_df.item_id.isin(cleaned_item_ids), 'condition_clean'] = 1
        extracted_feature_df.loc[extracted_feature_df.item_id.isin(static_item_ids), 'condition_clean'] = 2

        
    except:
        return
    
    # save images
    if save_img:
        save_img(cleaned_df)
    
    # save extracted features to db
    if save_fe:

        insert_extracted_feature(extracted_feature_df)

        
    if save_db:
        
        df_lst =[]

        for idx, group in cleaned_df.groupby('ITEM_ID'):
            try:
                df_lst.append(get_sell_amount_by_item_id(group))

            except:
                continue


        if len(df_lst) > 0:

            result = pd.concat(df_lst)
            result[['COLLECT_DAY']] = result.index
            insert_sell_amt(result)
#             result.to_sql(con=wspider_temp_engine, name='MWS_COLT_ITEM_SELL_AMT_DEV', if_exists='append')
#             logging.warning('done with %s' % str(file))
        



# In[14]:


ids_df = pd.read_sql_query("SELECT ID FROM MWS_COLT_ITEM WHERE RELEASE_DT > '2018-01-01'", wspider_engine)

DENOM = 1000
item_ids = ids_df.ID.values
n_batches = math.ceil( len(item_ids) / float(DENOM))
batch_ls = [str(tuple(batch)) for batch in np.array_split(item_ids, n_batches)]
batch_lst = [(idx, row) for idx, row in enumerate(batch_ls)]


# In[15]:


Parallel(n_jobs=-1)(map(delayed(process_full_batch), batch_lst))


# In[17]:


# DENOM = 50
# item_ids = ids_df.ID.values[:1000]
# n_batches = math.ceil( len(item_ids) / float(DENOM))
# batch_ls = [str(tuple(batch)) for batch in np.array_split(item_ids, n_batches)]
# batch_lst = [(idx, row) for idx, row in enumerate(batch_ls)]

# process_full_batch(batch_lst[5], save_db=True, save_fe=True)
# process_full_batch(batch_lst[6], save_db=True, save_fe=True)
# process_full_batch(batch_lst[7], save_db=True, save_fe=True)
# process_full_batch(batch_lst[8], save_db=True, save_fe=True)
# process_full_batch(batch_lst[9], save_db=True, save_fe=True)
# process_full_batch(batch_lst[10], save_db=True, save_fe=True)
# process_full_batch(batch_lst[11], save_db=True, save_fe=True)
# process_full_batch(batch_lst[12], save_db=True, save_fe=True)

