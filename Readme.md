* python3 first
* copy all *csv files into data folder
* install arboretum https://github.com/sh1ng/arboretum
* install lightgbm
* $ python create_products.py
* $ python split_data_set.py
* $ python orders_comsum.py
* $ python user_product_rank.py
* $ python create_prod2vec_dataset.py
* $ python skip_gram_train.py
* $ python skip_gram_get.py
* $ python arboretum_cv.py # optional just to see CV
* $ python lgbm_cv.py # optional...
* $ python arboretum_submition.py # prediction with arboretum
* $ python lgbm_submition.py # prediction with lgbm
* merge probabilities from 'data/prediction_arboretum.pkl' and 'data/prediction_lgbm.pkl'
* $ python f1_optimal.py 
* PROFIT!!!!!
