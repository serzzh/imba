
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

    labels = pd.read_pickle(os.path.join(path, 'chunk_0.pkl'))
    user_product = pd.read_pickle(os.path.join(path, 'previous_products.pkl'))

    order_comsum = pd.read_pickle('../input/orders_comsum.pkl')

    order_comsum = pd.merge(order_comsum, orders, on=['user_id', 'order_number'])[['user_id', 'order_number', 'days_since_prior_order_comsum', 'order_id']]

    order_product = pd.merge(order_prior, orders, on='order_id')[['order_id', 'product_id', 'eval_set']]
    order_product_train_test = labels[['order_id', 'product_id', 'eval_set']]

    order_product = pd.concat([order_product, order_product_train_test])

    order_product = pd.merge(order_product, order_comsum, on='order_id')

    print(order_product.columns)


    order_product = pd.merge(order_product, user_product, on=['user_id', 'product_id'])

    del orders
    del order_prior
    del order_product_train_test
    del labels
    del user_product
    del order_comsum
    


    
    ### My code
    temp = order_product.groupby(['user_id', 'product_id']).agg({'order_number': ['mean', 'max']})
    temp.columns = temp.columns.droplevel(0)
    temp.rename(columns={'max': 'order_number_max',
                         'mean': 'order_number_mean'}, inplace=True)
    temp = temp.reset_index()
    temp['order_number_skew'] = temp.order_number_mean / temp.order_number_max
    
    temp = temp.drop('order_number_max', axis=1)
    temp.to_pickle('../input/order_num_stat.pkl')
    del temp


    aggregated = pd.read_pickle('../input/product_period.pkl')
    aggregated['median'] = aggregated.periods.apply(lambda x: np.median(x))
    aggregated['mean'] = aggregated.periods.apply(lambda x: np.mean(x))
    aggregated['last'] = aggregated.periods.apply(lambda x: x[-1])
    aggregated['prev1'] = aggregated.periods.apply(lambda x: x[-2] if len(x) > 1 else 0)
    aggregated['prev2'] = aggregated.periods.apply(lambda x: x[-3] if len(x) > 2 else 0)

    aggregated.drop('periods', axis=1, inplace=True)
    aggregated['up_recency'] = aggregated['last'] / aggregated['mean']
    aggregated['up_recency2'] = aggregated['prev1'] / aggregated['mean']
    aggregated['up_recency3'] = aggregated['prev2'] / aggregated['mean']

    aggregated.to_pickle('../input/product_periods_stat.pkl')
    
       
    #temp = order_product.groupby('user_id').agg({'order_number': 'max'}).rename(columns={'order_number': 'user_order_last'})
    
    del aggregated
    
    
    ###Bayes code begins
    print ('users: ', len(order_product.user_id.unique()), ' products: ', len(order_product.product_id.unique()))
    order_product = order_product.loc[order_product.eval_set=='prior']
    

    order_product["user_n_orders"] = order_product.groupby('user_id')["order_number"].transform('max')
       
    
    order_product = order_product[['user_id', 'product_id', 'order_number', 'user_n_orders']].sort_values(
            ['user_id', 'product_id', 'order_number']).reset_index()
    order_product['x1'] = 1
    n_max_order = max(order_product['user_n_orders'])

    print ('users: ', len(order_product.user_id.unique()), ' products: ', len(order_product.product_id.unique()))


    enum = pd.DataFrame(columns=['order_number','user_n_orders'])    
    
    
    for n in xrange(n_max_order+1):
        Pmax = pd.DataFrame(np.column_stack((range(1,n+1), 
                                            #np.repeat(max(group.user_id), n),
                                            #np.repeat(max(group.product_id), n),
                                            np.repeat(n, n))),
                             columns=['order_number','user_n_orders'])
        enum = enum.append(Pmax)
    
    ### main part for function
    
    def bayes_df(df, enum):
        print ('users0: ', len(df.user_id.unique()), ' products: ', len(df.product_id.unique()))
        temp = df[['user_id', 'product_id', 'user_n_orders']].groupby(['user_id', 'product_id'
                            ]).first().reset_index()
        print ('users1: ', len(temp.user_id.unique()), ' products: ', len(temp.product_id.unique()))
        
        temp = temp.merge(enum, 'right', on=['user_n_orders'])
        print ('users2: ', len(temp.user_id.unique()), ' products: ', len(temp.product_id.unique()))
        
        temp.drop('user_n_orders', axis=1, inplace=True)
        temp = temp.sort_values(['user_id', 'product_id', 'order_number'])
        print ('users3: ', len(temp.user_id.unique()))
        k = df[['user_id', 'product_id', 'order_number','x1']]
        temp = temp.merge(k, 'left', on=['user_id', 'product_id', 'order_number'])
        print ('users4: ', len(k.user_id.unique()))
        print ('users5: ', len(temp.user_id.unique()))
        
        temp['x1'].fillna(0, inplace=True)
        temp['x1'] = temp['x1']*2 + temp['x1'].shift(-1)
        temp['max_ornum'] = temp.groupby(['user_id','product_id'])['order_number'].transform('max')
        temp = temp.loc[temp.order_number<temp.max_ornum,]
        print ('users6: ', len(temp.user_id.unique()))
        temp = temp.groupby(['user_id','product_id'])['x1'].value_counts().unstack(level=-1).reset_index()
        print ('users7: ', len(temp.user_id.unique()))
        return temp
    
    
    v = order_product['user_id'].unique()
    print ('users: ', len(order_product.user_id.unique()), ' products: ', len(order_product.product_id.unique()))

    n=5
    index = np.array_split(v, n)
    result = pd.DataFrame()
    for i in xrange(n):
        df = order_product.loc[np.in1d(order_product.user_id , index[i])]
        result = result.append(bayes_df(df, enum))
        print('finished: ', i)
        print('dim result:', result.shape)
        print ('users: ', len(result.user_id.unique()), ' products: ', len(result.product_id.unique()))

    del df
    print ('users: ', len(result.user_id.unique()), ' products: ', len(result.product_id.unique()))

    result.columns = [u'user_id', u'product_id', u'B0_0', u'B0_1', u'B1_0', u'B1_1']
    result.fillna(0,inplace=True)
    result.to_pickle('../input/bayes_backup.pkl')
    
    order_product["user_n_orders"] = order_product.groupby('user_id')["order_number"].transform('max')
    order_product["max_ornum"] = order_product.groupby(['user_id','product_id'])["order_number"].transform('max')
    order_product["is_last"] = 1*(order_product["user_n_orders"]==order_product["max_ornum"])
    result = result.merge(
            order_product[['user_id', 'product_id','is_last']].drop_duplicates(), 'inner', on=['user_id','product_id'])
    
    result['bayes_prob'] = np.where(result.is_last==1, result.B1_1/(result.B1_1+result.B1_0), result.B0_1/(result.B0_1+result.B0_0))
        
    result.to_pickle('../input/bayes_final.pkl')

    ##end of main part
