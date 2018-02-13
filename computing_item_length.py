
# coding: utf-8

# In[1]:


# %load_ext autoreload

# %autoreload 2


from joblib import Parallel, delayed
import time
import glob, os
from os import listdir
from util import *


# In[100]:


engine = create_engine("mysql://wspider:wspider00!q@133.186.143.65:3306/wspider", pool_size=20,
                       connect_args={'connect_timeout': 10000})
add_engine_pidguard(engine)
ids_df = pd.read_sql_query("SELECT ID FROM MWS_COLT_ITEM WHERE RELEASE_DT > '2017-11-01' AND RELEASE_DT < '2017-12-01'", engine)


# In[79]:


def mp_worker(batches):

    idx, query = batches
    
    logging.warning('wokring on %s' % str(idx))
    item_ivt_df = pd.read_sql_query("SELECT * FROM MWS_COLT_ITEM_IVT WHERE ITEM_ID in %s" % query, engine)
    item_ivt_df.to_pickle("data/pickle/ivt_item/ivt_item_%s.pkl" % str(idx))    
    logging.warning('done with %s' % str(idx))

def save_ivt_to_pickle(batch_lst, by=1000):
    Parallel(n_jobs=-1)(map(delayed(mp_worker), batch_lst))
        


# In[108]:


item_ids = ids_df.ID.values
n_batches = math.ceil( len(item_ids) / float(1000))
batch_ls = [str(tuple(batch)) for batch in np.array_split(item_ids, n_batches)]
batch_lst = [(idx, row) for idx, row in enumerate(batch_ls)]


# In[ ]:


save_ivt_to_pickle(batch_lst)

