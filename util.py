from sqlalchemy import create_engine
import pandas as pd
import math
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import cv2
import urllib
import matplotlib.image as mpimg
import seaborn as sns
from scipy import stats
import pymysql
import sys
from multiprocessing import Queue
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import qgrid
import logging
import pickle
import os
import warnings
import glob
from sqlalchemy import event
from sqlalchemy import exc


import itertools
from joblib import Parallel, delayed
import time
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from fancyimpute import KNN

import multiprocessing

reload(sys)
sys.setdefaultencoding('utf8')
sns.set(style="whitegrid", color_codes=True)
sns.set_style()
pd.set_option('display.height', 10)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


######## feature engineering #################

def get_label_from_dbscan(df, eps=0.2, min_samples=3, outlier=True):

    df = df.fillna(-1)
    outlier=True

    date = df.index
    df['INDEX'] = np.arange(3, len(df.STOCK_AMOUNT) + 3)
    Z = df[['STOCK_AMOUNT', 'INDEX']].values
    Z = np.vstack((Z, [[0, 2], [500, 1]]))

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    Z[:, 0] = scaler.fit_transform(Z[:, 0].reshape(-1, 1))[:, 0]
    X = StandardScaler().fit_transform(Z)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    return (n_clusters_, labels[:-2])


def get_feature_engineered_bundle(df):
    
    def get_arr_in_cluster(df):

        empty_lst = []
        for name, group in df.groupby('label')['STOCK_AMOUNT']:
            result_lst = np.sort(np.diff(group))[1:-1]
            empty_lst = np.append(empty_lst, result_lst)
        arr_in_cluster = -empty_lst[empty_lst < 0]
        return arr_in_cluster
    
    # The number of unique stock_id

    df = df.set_index("REG_DT")
    unique_stock_ids = df.STOCK_ID.unique()
    n_unique_stock_id = len(unique_stock_ids) 
    
    # select a single stock_id
    tmp2 = list(df.groupby('STOCK_ID'))[0][1]
    
    
    
    # The ratio of NA
    tmp3 = tmp2.resample('1D').first()
    
    # The number of days
    n_days = len(tmp3.ID)
    
    if n_days <=1:
        return
    
    null_arr = pd.isnull(tmp3.ID).values
    ratio_of_na = sum(null_arr) / float(n_days)
    
    
    consecutive_lst = [ sum( 1 for _ in group ) for key, group in itertools.groupby( null_arr ) if key ]
    
    
    # The max value of consecutive NAs 
    max_consecutive_na = max([0] + consecutive_lst)
    

    # The instances of consecutive NAs
    n_consecutive_na = len(consecutive_lst)
    
    # Define a stock array
    stock_arr = tmp3.STOCK_AMOUNT.values
    
    # The medain
    median_v = np.nanmedian(stock_arr)
    
    # Std
    std_v = np.nanstd(stock_arr)
    
    # max, min
    max_v = np.nanmax(stock_arr)
    min_v = np.nanmin(stock_arr)
    
    # The range between max and min
    range_v = max_v - min_v
    
    stock_na_removed = stock_arr[~np.isnan(stock_arr)]
    
    consecutive_same_lst = [ sum( 1 for _ in group ) for key, group in itertools.groupby( stock_na_removed ) if key ]
    
    if len(consecutive_same_lst) == 0:
        ratio_same_value = 0
    else:
        ratio_same_value = max(consecutive_same_lst) / float(n_days)
    
    
    n_jumps = sum(np.diff(stock_na_removed) > 0)
    max_drop = -min(np.diff(stock_na_removed))
    
    tmp3['STOCK_AMOUNT'] = tmp3.STOCK_AMOUNT.replace(np.nan, -1)
    n_cluster, label = get_label_from_dbscan(tmp3)
    
    tmp3['label'] = label
    arr_in_cluster = get_arr_in_cluster(tmp3)
    
    mean_in_cluster = np.nanmean(arr_in_cluster)
    std_in_cluster = np.nanstd(arr_in_cluster)
    

    
    bundle = {
        'item_id': df.ITEM_ID.values[0],
        'stock_id': tmp3.STOCK_ID.values[0],
        'n_unique_stock_id': n_unique_stock_id,
        'n_days': n_days,
        'ratio_of_na': ratio_of_na,
        'max_consecutive_na': max_consecutive_na,
        'n_consecutive_na': n_consecutive_na,
        'median_v': median_v,
        'std_v': std_v,
        'max_v': max_v,
        'ratio_drop': max_drop / float(max_v),
        'min_v': min_v,
        'range_v': range_v,
        'ratio_same_value': ratio_same_value,
        'n_jumps': n_jumps,
        'max_drop': max_drop,
        'n_cluster': n_cluster,
        'mean_in_cluster': mean_in_cluster,
        'std_in_cluster': std_in_cluster

    }
    
    return bundle

def get_filtered_fg_df(feature_engineered_df):
    item_ids_static = feature_engineered_df.item_id[(feature_engineered_df.std_in_cluster == 0.0)].values
    data_df_cleaned = feature_engineered_df[feature_engineered_df.mean_in_cluster.notnull()]
    purified_df = data_df_cleaned[(data_df_cleaned.ratio_drop < 0.2)
                          & (data_df_cleaned.ratio_same_value < 0.3)
                          & (data_df_cleaned.n_jumps < 2)
                          & (data_df_cleaned.n_days > 20)
                          & (data_df_cleaned.std_in_cluster > 0.2)
                          & (data_df_cleaned.std_in_cluster < 4)
                          & (data_df_cleaned.ratio_of_na < 0.5)
                          & (data_df_cleaned.n_unique_stock_id < 10)]
    return purified_df



def get_ivt_item(item_id):
    result = list(data_dict[item_id].groupby('STOCK_ID'))[0][1]
    return result


def map_clean_up_target_df(series):

    tmp_df = pd.DataFrame(series)
    tmp_df.columns = ['STOCK_AMOUNT']
    tmp_df['REG_DT'] = tmp_df.index
    return clean_up_target_df(tmp_df)['sell_impute']


def get_sell_amount_by_item_id(df):
    df_pivot = df.pivot_table(index='REG_DT', columns='STOCK_ID', values='STOCK_AMOUNT')
    sell_amount_by_stock = df_pivot.apply(map_clean_up_target_df)
    sell_amount_total = sell_amount_by_stock.sum(axis=1)
    
    result = pd.DataFrame(sell_amount_total)
    result.columns = ['SELL_AMOUNT']
    item_id = df.ITEM_ID.values[0]
    result['ITEM_ID'] = item_id
    result['REG_ID'] = 'SERVER'
    result['UPT_DT'] = pd.to_datetime('now')
    result['COLLECT_DAY'] = pd.to_datetime('now')
    result['UPT_ID'] = 'FILTER ALGO'
    
    
    return result


def clean_up_target_df(target):


    n_cluster, label = get_label_from_dbscan(target, eps=0.15, min_samples=3)
    target['label'] = label
    target = target[['STOCK_AMOUNT', 'label', 'REG_DT']]
    labels = target.label.unique()
    target = target.set_index('REG_DT')
    target = target.resample('1D').first()
    target['STOCK_AMOUNT_imputed'] = target['STOCK_AMOUNT']

    

    for label in labels:
        idx = np.where(target.label.values == label)[0]
        if len(idx) == 0:
            continue
        start_v = min(idx)
        end_v = max(idx)
        target.loc[start_v:end_v+1, 'STOCK_AMOUNT_imputed'] = target['STOCK_AMOUNT'][start_v:end_v+1].interpolate(method='from_derivatives')

    target['STOCK_AMOUNT_imputed'] = target['STOCK_AMOUNT'].interpolate(method='from_derivatives')

    target['STOCK_AMOUNT_imputed'] = target.STOCK_AMOUNT_imputed.round()
    target['weekday_name'] = target.index.dayofweek
    target['sell'] = np.append([0], np.negative(np.diff(target.STOCK_AMOUNT_imputed)))
    target.loc[target['sell'].values < 0, 'sell'] = np.nan
    target.sell.astype(float)
    target['zscore'] = np.abs(target.sell - target.sell.mean() / max(0.0001, target.sell.std()))
    target.loc[target['zscore'] > 4, 'sell'] = np.nan
    X_incomplete = target[['sell', 'weekday_name']].values

    try:
        X_filled_knn = KNN(k=1).complete(X_incomplete)
        target['sell_impute'] = X_filled_knn[:,0]
    except:
        target['sell_impute'] = target['sell']
        
    target['STOCK_AMOUNT_imputed_trimed'] = target['STOCK_AMOUNT_imputed']
    
    cond = np.append([0], np.negative(np.diff(target.STOCK_AMOUNT_imputed))) < 0
    
    target.loc[cond, 'STOCK_AMOUNT_imputed_trimed'] = np.nan

    return target
    


# common functions
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


def get_itemId_from_goodsNum(goodsNum):
    sites = {1: 'GSSHOP', 2: 'lotte.com', 3: 'HMALL'}
    list_itemId = []

    conn = pymysql.connect(host='133.186.134.155', user='wspidermr', password='wspidermr00!q', db='lf-bigdata-real-5',
                           charset='utf8')
    curs = conn.cursor(pymysql.cursors.DictCursor)
    queryString = "SELECT SITE_NO, ITEM_NUM FROM MLF_GOODS where GOODS_NUM = '" + goodsNum + "';"
    curs.execute(queryString)
    df_itemNum = pd.DataFrame(curs.fetchall())
    conn.close()

    conn = pymysql.connect(host='133.186.143.65', user='wspider', password='wspider00!q', db='wspider', charset='utf8')
    curs = conn.cursor()
    for i, r in df_itemNum.iterrows():
        queryString = "SELECT ID FROM wspider.MWS_COLT_ITEM WHERE SITE_NAME='" + sites[
            r.SITE_NO] + "' and ITEM_NUM = '" + r.ITEM_NUM + "';"
        curs.execute(queryString)
        df_itemId = pd.DataFrame(list(curs.fetchall()), columns=['ITEM_ID'])
        list_itemId += df_itemId.ITEM_ID.tolist()
    conn.close()

    return list_itemId


def get_tables_from_goodsNum(item_id):

    engine = get_engine()

    colt_item_df = pd.read_sql_query("SELECT * FROM MWS_COLT_ITEM WHERE ID = %s" % item_id, engine)

    colt_item_ivt_df = pd.read_sql_query("SELECT * FROM MWS_COLT_ITEM_IVT WHERE ITEM_ID = %s" % item_id, engine)
    colt_item_ivt_df = colt_item_ivt_df.set_index('REG_DT')

    colt_item_image_df = pd.read_sql_query("SELECT * FROM MWS_COLT_IMAGE WHERE ITEM_ID = %s" % item_id, engine)

    image_url = colt_item_image_df['GOODS_IMAGE'][0]
    image = url_to_image(image_url)

    return {
        'colt_item_df': colt_item_df,
        'colt_item_ivt_df': colt_item_ivt_df,
        'image': image

    }


def run_dbscan(df, eps=0.2, min_samples=3, outlier=True):
    date = df.index
    df['INDEX'] = np.arange(3, len(df.STOCK_AMOUNT) + 3)
    Z = df[['STOCK_AMOUNT', 'INDEX']].values
    Z = np.vstack((Z, [[0, 2], [500, 1]]))

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    Z[:, 0] = scaler.fit_transform(Z[:, 0].reshape(-1, 1))[:, 0]

    # scaler = preprocessing.Normalizer()
    # X = scaler.fit_transform(Z)

    X = StandardScaler().fit_transform(Z)


    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = Z[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=3)

        if outlier:
            xy = Z[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=10)

    plt.title('Estimated number of clusters: %d' % n_clusters_)


#     plt.show()
def plot_all_from_goodsNum(item_id, engine):
    engine = get_engine()
    item_ids = get_itemId_from_goodsNum(item_id)
    
    for item_id in item_ids:
        print(item_id)
    

def plot_sample_from_item_ivts(df):
    df = df.set_index('REG_DT')
    plt.figure(figsize=(20, 5))

    unqiue_stock = df.STOCK_ID.unique()
    n_unqiue_stock = len(unqiue_stock)

    item_id = df.ITEM_ID.values[0]

    plt.subplot(1, n_unqiue_stock+1, 1)
    plt.imshow(get_image_from_item_id(item_id))
    for i in range(n_unqiue_stock):
        stock_id_group = df[df.STOCK_ID == unqiue_stock[i]]
        plt.subplot(1, n_unqiue_stock+1, 2 + i)
        plt.title(str(item_id) + "-" + str(stock_id_group.STOCK_ID.values[0]))
        plt.scatter(x = stock_id_group.index, y = stock_id_group.STOCK_AMOUNT, s=10)
        plt.xticks(rotation="60")
    plt.show()

def get_engine():
    engine = create_engine("mysql://wspider:wspider00!q@133.186.143.65:3306/wspider",
                           connect_args={'connect_timeout': 10000})

    return engine

def get_image_from_item_id(item_id):
    engine = get_engine()

    colt_item_image_df = pd.read_sql_query("SELECT * FROM MWS_COLT_IMAGE WHERE ITEM_ID = %s" % item_id, engine)
    image_url = colt_item_image_df['GOODS_IMAGE'][0]
    image = url_to_image(image_url)
    return image


def plot_sample_by_goodsNum(goodsNum, engine):
    goods_num = get_itemId_from_goodsNum(goodsNum)
    data = get_tables_from_goodsNum(goods_num[0], engine)
    data['colt_item_ivt_df'] = data['colt_item_ivt_df']['2017-11-01':][['OPTION', 'STOCK_AMOUNT', 'STOCK_ID', 'SIZE_OPTION']]

    lst = data['colt_item_ivt_df'].STOCK_ID.unique()

    plt.figure(figsize=(15, len(lst) * 5))

    index = 1

    n_min = max(3, len(lst))

    for stock_id in lst[:n_min]:
        tmp2 = data['colt_item_ivt_df'][data['colt_item_ivt_df'].STOCK_ID == stock_id]

        # original
        plt.subplot(len(lst), 3, index)
        tmp2['STOCK_AMOUNT'] = tmp2['STOCK_AMOUNT'].replace(np.nan, None)
        plt.scatter(x=tmp2.index, y=tmp2.STOCK_AMOUNT, s=10)
        plt.xticks(rotation='60')
        plt.title(tmp2[tmp2.STOCK_ID == stock_id].OPTION.values[0])
        index += 1

        # clustering (original)
        plt.subplot(len(lst), 3, index)
        run_dbscan(tmp2)
        index += 1

        # imputed
        plt.subplot(len(lst), 3, index)

        tmp = tmp2.resample('3H').first()
        tmp["STOCK_AMOUNT"] = tmp["STOCK_AMOUNT"].replace(0, None)
        tmp["STOCK_AMOUNT"] = tmp.STOCK_AMOUNT.interpolate(method='linear').values
        selected = np.append([0], np.negative(np.diff(tmp["STOCK_AMOUNT"])))
        mask = (selected < 0) | (np.abs(stats.zscore(selected)) > 1)
        tmp["STOCK_AMOUNT"][mask] = None
        #         tmp["STOCK_AMOUNT"] = tmp["STOCK_AMOUNT"].interpolate(methoed='nearest', limit_direction='').values

        plt.scatter(tmp.index, tmp.STOCK_AMOUNT, s=10)
        index += 1
    plt.show()


#         tmp = tmp[['STOCK_AMOUNT']]

#         tmp = tmp.dropna()

#         # clustering (imputed)
#         plt.subplot(len(lst), 4, index)
#         run_dbscan(tmp, eps=0.1, min_samples=4, outlier=False)
#         index += 1

def plot_image(img):
    plt.imshow(img)


def examine_goods(goods_num, plot=True):
    engine = create_engine("mysql://wspider:wspider00!q@133.186.143.65:3306/wspider",
                           connect_args={'connect_timeout': 10000})
    sample_goods_num = goods_num
    goods_nums = get_itemId_from_goodsNum(sample_goods_num)

    bundle_lst = []

    for goods_num in goods_nums:

        data = get_tables_from_goodsNum(goods_num, engine)
        data['colt_item_ivt_df'] = data['colt_item_ivt_df']['2017-11-01':][['OPTION', 'STOCK_AMOUNT', 'STOCK_ID', 'SIZE_OPTION']]

        bundle_lst.append(data)

        if plot:
            plot_image(data['image'])
            plot_sample_by_goodsNum(sample_goods_num, engine)

    return bundle_lst


def save_item_ivts(engine, item_ids, save=True):

    output_lst = []

    def mp_worker(item_id):

        sample_item_900 = pd.read_sql_query("SELECT * FROM MWS_COLT_ITEM_IVT WHERE ITEM_ID = %s" % item_id, engine)
        item_id = sample_item_900.ITEM_ID[0]
        logging.warning(item_id)
        return ({
            'item_id': item_id,
            'df': sample_item_900
        })

    p = multiprocessing.Pool(8)
    results = [p.apply_async(mp_worker, args=(x,)) for x in item_ids]

    for i in range(len(results)):

        try:

            result = results[i].get()
            output_lst.append(result)
        except:
            continue

    if save:
        pickle.dump(output_lst, open("data/MWS_COLT_ITEM_IVT.pkl", "wb"))

    return output_lst

def get_item_ivts():
    return pickle.load(open("data/pickle/MWS_COLT_ITEM_IVT.pkl", "rb"))

def add_engine_pidguard(engine):
    """Add multiprocessing guards.

    Forces a connection to be reconnected if it is detected
    as having been shared to a sub-process.

    """

    @event.listens_for(engine, "connect")
    def connect(dbapi_connection, connection_record):
        connection_record.info['pid'] = os.getpid()

    @event.listens_for(engine, "checkout")
    def checkout(dbapi_connection, connection_record, connection_proxy):
        pid = os.getpid()
        if connection_record.info['pid'] != pid:
            # substitute log.debug() or similar here as desired
            warnings.warn(
                "Parent process %(orig)s forked (%(newproc)s) with an open "
                "database connection, "
                "which is being discarded and recreated." %
                {"newproc": pid, "orig": connection_record.info['pid']})
            connection_record.connection = connection_proxy.connection = None
            raise exc.DisconnectionError(
                "Connection record belongs to pid %s, "
                "attempting to check out in pid %s" %
                (connection_record.info['pid'], pid)
            )