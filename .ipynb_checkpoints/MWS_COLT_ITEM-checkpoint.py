
# coding: utf-8


from sqlalchemy import create_engine
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import urllib
import matplotlib.image as mpimg
import seaborn as sns
from scipy import stats
import pymysql
import sys  
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler



reload(sys)  
sys.setdefaultencoding('utf8')
sns.set(style="whitegrid", color_codes=True)
sns.set_style()
pd.set_option('display.height', 3000)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# common functions


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
    # return the image
    return image


def get_itemId_from_goodsNum(goodsNum) :
    sites = { 1:'GSSHOP', 2:'lotte.com', 3:'HMALL' }
    list_itemId = []


    conn = pymysql.connect(host='133.186.134.155', user='wspidermr', password='wspidermr00!q', db='lf-bigdata-real-5', charset='utf8')
    curs = conn.cursor(pymysql.cursors.DictCursor)
    queryString = "SELECT SITE_NO, ITEM_NUM FROM MLF_GOODS where GOODS_NUM = '"+goodsNum+ "';"
    curs.execute(queryString)
    df_itemNum = pd.DataFrame( curs.fetchall() )
    conn.close()


    conn = pymysql.connect(host='133.186.143.65', user='wspider', password='wspider00!q', db='wspider', charset='utf8')
    curs = conn.cursor()
    for i, r in df_itemNum.iterrows():
        queryString = "SELECT ID FROM wspider.MWS_COLT_ITEM WHERE SITE_NAME='"+sites[r.SITE_NO]+"' and ITEM_NUM = '"+r.ITEM_NUM+"';"
        curs.execute(queryString)
        df_itemId = pd.DataFrame(list(curs.fetchall()), columns=['ITEM_ID'])
        list_itemId += df_itemId.ITEM_ID.tolist()
    conn.close()


    return list_itemId

def get_tables_from_goodsNum(item_id, engine):
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

def run_dbscan(df, eps=0.7, min_samples=3):
    
    df['INDEX'] = np.arange(1, len(df.STOCK_AMOUNT) + 1)
    X = df[['STOCK_AMOUNT', 'INDEX']].values
    X = StandardScaler().fit_transform(X)


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
    
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=3)
    
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10)
    
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    
def plot_sample_by_goodsNum(goodsNum):
    
    goods_num = get_itemId_from_goodsNum(goodsNum)
    data = get_tables_from_goodsNum(goods_num[0], engine)
    data['colt_item_ivt_df'] = data['colt_item_ivt_df'][['OPTION', 'STOCK_AMOUNT', 'STOCK_ID', 'SIZE_OPTION']]

    lst = data['colt_item_ivt_df'].STOCK_ID.unique()

    plt.figure(figsize=(10, len(lst) * 5))
    for index, stock_id in enumerate(lst):
        tmp2 = data['colt_item_ivt_df'][data['colt_item_ivt_df'].STOCK_ID == stock_id]
        plt.subplot(len(lst), 1, index + 1)
        plt.scatter(x= tmp2.index, y = tmp2.STOCK_AMOUNT, s = 10)
        plt.title(tmp2[tmp2.STOCK_ID == stock_id].OPTION.values[0])
        
    plt.figure(figsize=(10, len(lst) * 5))
    for index, stock_id in enumerate(lst):
        tmp2 = data['colt_item_ivt_df'][data['colt_item_ivt_df'].STOCK_ID == stock_id]
        run_dbscan(tmp2)
# connection to db


engine = create_engine("mysql://wspider:wspider00!q@133.186.143.65:3306/wspider", connect_args={'connect_timeout': 10000})

# sample item = 2-436849581
goods_num = get_itemId_from_goodsNum('1-27708159')
data = get_tables_from_goodsNum(goods_num[0], engine)
data['colt_item_ivt_df'] = data['colt_item_ivt_df'][['OPTION', 'STOCK_AMOUNT', 'STOCK_ID', 'SIZE_OPTION']]

lst = data['colt_item_ivt_df'].STOCK_ID.unique()

#array(['-577362230978209480', '-577362230978209479', '-577362230978209473',
 #      '-808140771723420265', '2623289634502587079',
  #     '-8344094660088529121', '2623289634502587073', '2623289634502587072'], dtype=object)


#lst = ['-577362230978209480', '-577362230978209479', '-577362230978209473', 
#       '-808140771723420265', '2623289634502587079', '-8344094660088529121',
#       '2623289634502587073', '2623289634502587072']
plt.figure(figsize=(10, len(lst) * 5))
for index, stock_id in enumerate(lst):
    tmp2 = data['colt_item_ivt_df'][data['colt_item_ivt_df'].STOCK_ID == stock_id]
    plt.subplot(len(lst), 1, index + 1)
    plt.scatter(x= tmp2.index, y = tmp2.STOCK_AMOUNT)
    plt.title(tmp[tmp.STOCK_ID == stock_id].OPTION.values[0])



#### plot sample using dbscan

plot_sample_by_goodsNum('1-27861713')






plt.imshow(data['image'])



lst = ['2623289634502587073']
plt.figure(figsize=(10, len(lst) * 5))
for index, stock_id in enumerate(lst):
    tmp2 = data['colt_item_ivt_df'][data['colt_item_ivt_df'].STOCK_ID == stock_id]
#    plt.subplot(len(lst), 1, index + 1)
    plt.scatter(x= tmp2.index, y = tmp2.STOCK_AMOUNT)
 #   plt.title(tmp[tmp.STOCK_ID == stock_id].OPTION.values[0])

tmp1 = data['colt_item_ivt_df'][data['colt_item_ivt_df'].STOCK_ID == lst[1]]
plt.scatter(x= tmp1.index, y = tmp1.STOCK_AMOUNT)

tmp2 = data['colt_item_ivt_df'][data['colt_item_ivt_df'].STOCK_ID == '2623289634502587073']
tmp3 = pd.concat([tmp1, tmp2])

tmp4 = tmp3['2017-10':'2017-12']


plt.scatter(x= tmp1.index, y = tmp1.STOCK_AMOUNT)
run_dbscan(tmp1, 0.6, 3)









plt.scatter(tmp4.index, tmp4.STOCK_AMOUNT)
tmp4['STOCK_AMOUNT_MA5'] = tmp4['STOCK_AMOUNT'].rolling(window = 10).mean()
plt.scatter(tmp4.index, tmp4.STOCK_AMOUNT_MA5)


(np.abs(stats.zscore(tmp4['STOCK_AMOUNT'])) < 3)




np.append([0], np.abs(np.diff(tmp4['STOCK_AMOUNT'].values)))

###

plt.scatter(tmp4.index, tmp4.STOCK_AMOUNT)

tmp5 = tmp4.resample('D').first()

import math

sns.swarmplot(tmp4.index, tmp4.STOCK_AMOUNT)


#tmp5['STOCK_AMOUNT'] = tmp5['STOCK_AMOUNT'].interpolate(method='time')

#plt.scatter(x= tmp5.index, y = tmp5.STOCK_AMOUNT)

tmp5['STOCK_AMOUNT_ADJ'] = np.append([0], np.abs(np.diff(tmp5['STOCK_AMOUNT'].values)))


tmp5['WEEKDAY'] = tmp5.index.to_series().dt.weekday_name

sns.boxplot(x='WEEKDAY', y='STOCK_AMOUNT_ADJ', data=tmp5, order =['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.plot(tmp5.index, tmp5.STOCK_AMOUNT_ADJ)
plt.plot(tmp5.index, tmp5.WEEKDAY, color = 'red', linestyle='none', marker='.')






# look at -577362230978209473
tmp3 = tmp[tmp.STOCK_ID == '-577362230978209473']
tmp3.head(1)





plt.figure(figsize=(20, 10))
plt.scatter(x = tmp3.index, y = tmp3.STOCK_AMOUNT)





tmp4 = tmp3.resample('D').first()





tmp4.head()





tmp4['STATUS'] = np.where(tmp4.STOCK_AMOUNT.isnull(), "abnormal", "normal")





tmp4['STOCK_AMOUNT_ADJ'] = tmp4['STOCK_AMOUNT'].interpolate('time')





tmp4['STOCK_AMOUNT'] = tmp4['STOCK_AMOUNT'].fillna(0)





tmp4.head()





plt.figure(figsize=(20, 10))
select_normal_df = tmp4['2017-11-18':'2017-12-16'][tmp4.STATUS == 'normal']
plt.scatter(x = select_normal_df.index, y = select_normal_df.STOCK_AMOUNT)

select_abnormal_df = tmp4['2017-11-18':'2017-12-16'][tmp4.STATUS == 'abnormal']
plt.scatter(x = select_abnormal_df.index, y = select_abnormal_df.STOCK_AMOUNT, c='red')

# plt.scatter(x = tmp4[tmp4.STATUS == 'abnormal'].index, y = tmp4[tmp4.STATUS == 'abnormal'].STOCK_AMOUNT, c='red')





plt.figure(figsize=(20, 10))
select_abnormal_df = tmp4['2017-11-18':'2017-12-16'][tmp4.STATUS == 'normal']
plt.scatter(x = select_abnormal_df.index, y = select_abnormal_df.STOCK_AMOUNT_ADJ, c='black')

select_abnormal_df = tmp4['2017-11-18':'2017-12-16'][tmp4.STATUS == 'abnormal']
plt.scatter(x = select_abnormal_df.index, y = select_abnormal_df.STOCK_AMOUNT_ADJ, c='red')





tmp4['2017-11-18':'2017-12-16']['STOCK_AMOUNT']





stats.zscore(tmp4['2017-11-18':'2017-12-16']['STOCK_AMOUNT'])





(np.abs(stats.zscore(tmp4['2017-11-18':'2017-12-16']['STOCK_AMOUNT'])) < 3)





tmp4['2017-11-18':'2017-12-16'][(np.abs(stats.zscore(tmp4['2017-11-18':'2017-12-16']['STOCK_AMOUNT'])) > 3)]





tmp4.head()





plt.figure(figsize=(20, 10))
select_normal_df = tmp4['2017-11-18':'2017-12-16'][tmp4.STATUS == 'normal']
plt.scatter(x = select_normal_df.index, y = select_normal_df.STOCK_AMOUNT)

select_abnormal_df = tmp4['2017-11-18':'2017-12-16'][tmp4.STATUS == 'abnormal']
plt.scatter(x = select_abnormal_df.index, y = select_abnormal_df.STOCK_AMOUNT, c='red')
# plt.scatter(x = tmp4[tmp4.STATUS == 'abnormal'].index, y = tmp4[tmp4.STATUS == 'abnormal'].STOCK_AMOUNT, c='red')





# tmp[tmp.STOCK_ID == '6204056055357454558']['2018-1':'2018-2']





goods_num = get_itemId_from_goodsNum('AJ8153I')





data = get_tables_from_goodsNum(goods_num[0])





data['colt_item_ivt_df']





tmp2 = tmp[tmp.STOCK_ID == '-577362230978209479']





plt.scatter(x= tmp2.index, y = tmp2.STOCK_AMOUNT)





lst = []
for stock_id in data['colt_item_ivt_df']['STOCK_ID'].unique():
    tmp_df = data['colt_item_ivt_df'].copy()
    tmp_df = tmp_df[tmp_df.STOCK_ID == stock_id]
    tmp_df['SOLD_AMOUNT'] = np.abs(np.append([0], np.diff(tmp_df.STOCK_AMOUNT)))
    sold_amount = tmp_df['SOLD_AMOUNT'].resample('D').sum().fillna(0)
    lst.append(sold_amount)





data['colt_item_ivt_df']['STOCK_ID'].unique()





lst[0]['2017-10-10':'2017-10-26'] + lst[1]['2017-10-10':'2017-10-26'] + lst[2]['2017-10-10':'2017-10-26']





(lst[0] + lst[1] + lst[2] + lst[3]).plot()





data['colt_item_ivt_df']['2018-01-17':]





(lst[4] + lst[5] + lst[6] + lst[7]).plot()





sold_mount_total = np.sum(lst)





sold_mount_total





sold_mount_total.plot()





# tmp_df = df_item_ivt_overview[['STOCK_AMOUNT', 'STOCK_ID', 'REG_DT', 'OPTION', 'COLOR_OPTION', 'SIZE_OPTION']]





tmp_black_df = tmp_df[tmp_df.COLOR_OPTION == '블랙']
a = sns.tsplot(tmp_black_df, time='REG_DT', value='STOCK_AMOUNT', unit='OPTION', condition='OPTION')





tmp_white_df = tmp_df[tmp_df.COLOR_OPTION == '화이트']
a = sns.tsplot(tmp_white_df, time='REG_DT', value='STOCK_AMOUNT', unit='OPTION', condition='OPTION')





tmp_black_df.head()


####################### experiment

# tmp4 = tmp3.resample('D').first()








