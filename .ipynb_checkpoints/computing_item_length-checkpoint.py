{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [],
   "source": [
    "# %load_ext autoreload\n",
    "\n",
    "# %autoreload 2\n",
    "\n",
    "\n",
    "from joblib import Parallel, delayed\n",
    "import time\n",
    "import glob, os\n",
    "from os import listdir\n",
    "from util import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "engine = create_engine(\"mysql://wspider:wspider00!q@133.186.143.65:3306/wspider\",\n",
    "                       connect_args={'connect_timeout': 10000})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ids = pd.read_sql_query(\"SELECT ID FROM MWS_COLT_ITEM WHERE RELEASE_DT > '2018-01-01' AND SITE_NAME IN ('GSSHOP', 'HMALL') LIMIT 1000\", engine)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [],
   "source": [
    "engine = create_engine(\"mysql://wspider:wspider00!q@133.186.143.65:3306/wspider\",\n",
    "                           connect_args={'connect_timeout': 10000})\n",
    "add_engine_pidguard(engine)\n",
    "ids_df = pd.read_sql_query(\"SELECT ID FROM MWS_COLT_ITEM WHERE RELEASE_DT > '2017-11-01' AND RELEASE_DT < '2017-12-01'\", engine)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# colt_item_df = pd.read_sql_query(\"SELECT * FROM MWS_COLT_ITEM LIMIT 10\", engine)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [],
   "source": [
    "def mp_worker(item_id):\n",
    "\n",
    "    item_ivt_df = pd.read_sql_query(\"SELECT * FROM MWS_COLT_ITEM_IVT WHERE ITEM_ID = %s\" % str(item_id), engine)\n",
    "    bundle = {\n",
    "        'item_id': item_id,\n",
    "        'df': item_ivt_df,\n",
    "    }\n",
    "    \n",
    "    return bundle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def save_ivt_to_pickle(item_ids, by=1000):\n",
    "    start = 0\n",
    "    by = by\n",
    "    end = len(item_ids)\n",
    "\n",
    "    for idx in np.arange(start, end, by):\n",
    "        logging.warning( str((idx / float(end)) * 100) + \"% done\" )\n",
    "        results = Parallel(n_jobs=-1)(map(delayed(mp_worker), ids[idx:idx+by]))\n",
    "        result_df = pd.DataFrame(results)\n",
    "        result_df.to_pickle(\"data/pickle/ivt_item/ivt_item_%s-%s.pkl\" % (idx, idx+by-1))\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "metadata": {},
   "outputs": [],
   "source": [
    "ids = ids_df.ID.values\n",
    "save_ivt_to_pickle(ids, by=5000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {},
   "outputs": [],
   "source": [
    "# # def save_item_ivts(engine, item_ids, save=True):\n",
    "\n",
    "\n",
    "\n",
    "# output_lst = []\n",
    "\n",
    "# def mp_worker(item_id):\n",
    "\n",
    "#     sample_item_900 = pd.read_sql_query(\"SELECT * FROM MWS_COLT_ITEM_IVT WHERE ITEM_ID = %s\" % item_id, engine)\n",
    "#     item_id = sample_item_900.ITEM_ID[0]\n",
    "#     logging.warning(item_id)\n",
    "#     return ({\n",
    "#         'item_id': item_id,\n",
    "#         'df': sample_item_900,\n",
    "#         'n_row': len(sample_item_900)\n",
    "#     })\n",
    "\n",
    "# p = multiprocessing.Pool(8)\n",
    "# results = [p.apply_async(mp_worker, args=(x,)) for x in item_ids]\n",
    "\n",
    "# for i in range(len(results)):\n",
    "\n",
    "#     try:\n",
    "\n",
    "#         result = results[i].get()\n",
    "#         output_lst.append(result)\n",
    "#     except:\n",
    "#         continue\n",
    "\n",
    "# #     if save:\n",
    "# #         pickle.dump(output_lst, open(\"data/MWS_COLT_ITEM_IVT.pkl\", \"wb\"))\n",
    "\n",
    "# # return output_lst"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "for item in output_lst:\n",
    "    if item['n_row'] > 200:\n",
    "        print(str(item['item_id']) + \" - \" + str(item['n_row']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {},
   "outputs": [],
   "source": [
    "result_df = pd.concat([pd.read_pickle(pkl) for pkl in glob.glob('data/pickle/ivt_item/*.pkl')])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
