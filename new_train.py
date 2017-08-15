

import pandas as pd
import numpy as np
import os



if __name__ == '__main__':
    path = "../input"

    order_prior = pd.read_csv(os.path.join(path, "order_products__prior.csv"), dtype={'order_id': np.uint32,
                                                                                      'product_id': np.uint16,
                                                                                      'add_to_cart_order': np.uint8,
                                                                                      'reordered': bool})
    orders = pd.read_csv(os.path.join(path, "orders.csv"), dtype={'order_id': np.uint32,
                                                                  'user_id': np.uint32,
                                                                  'eval_set': 'category',
                                                                  'order_number': np.uint8,
                                                                  'order_dow': np.uint8,
                                                                  'order_hour_of_day': np.uint8
                                                                  })
    order_train = pd.read_csv(os.path.join(path, "order_products__train.csv"), dtype={'order_id': np.uint32,
                                                                                      'product_id': np.uint16,
                                                                                      'add_to_cart_order': np.uint8,
                                                                                      'reordered': bool})
    
    ##Find list of test users
    
    n_ord_train = order_train.order_id.unique()
    
    users_train = orders.loc[np.in1d(orders.order_id, n_ord_train)].user_id.unique()
    users_test = orders.loc[~np.in1d(orders.user_id, users_train)].user_id.unique()
    
    ## Find last orders for test dataset in prior
    
    orders2 = orders.loc[np.in1d(orders.user_id, users_test)]  
    orders2 = orders2.loc[orders.eval_set=='prior'].copy()

    
    
    orders2["user_n_orders"] = orders2.groupby('user_id')["order_number"].transform('max')
    orders2["is_last"] = 1*(orders2["user_n_orders"]==orders2["order_number"])   
    orders2_ids = orders2.loc[orders2.is_last==1].order_id
    
    
    
    
    ## Mark them as train
    
    orders.loc[np.in1d(orders.order_id, orders2_ids),'eval_set'] = 'train'
    orders2 = orders[np.in1d(orders.order_id, orders2_ids)]
       
    ## get order_train2 from order_prior
    
    order_train2 = pd.merge(order_prior, orders2, on='order_id')[order_prior.columns]
    order_prior2 = order_prior.loc[~np.in1d(order_prior.order_id, orders2_ids)]
    ## Collect all product_ids and create order_train2
    
    order_train2 = order_train.append(order_train2)
    
    
    orders.to_csv("orders2.csv")
    order_prior2.to_csv("order_products__prior2.csv")
    order_train2.to_csv("order_products__train2.csv")
    
    
    order_train2['reordered'].fillna(0, inplace=True)
    ## Add order_train2 to order_train
