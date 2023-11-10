import numpy as np
import json
from frank_wolfe_lc_mnl import FrankWolfeMNLMixEst

np.random.seed(1607)
# parse input data
def read_sample_data(in_sample=True):
    # replace by path to input file
    json_data = open('test_instance.dt').read()
    input_json = json.loads(json_data)
    products = np.array(input_json['products'])
    num_prods = len(products)
    offersets = dict()
    sales = np.array([])
    membership = np.array([])
    # determine whether to read in-sample or out-of-sample transactions
    transactions_to_parse = 'in_sample_transactions' if in_sample else 'out_of_sample_transactions'
    for transaction in input_json['transactions'][transactions_to_parse]:
        offered_products = np.array(transaction['offered_products'])
        chosen_product = transaction['product']
        # form the offer-set as a 1d array
        o_set_arr = np.zeros(num_prods, dtype=int)
        o_set_arr[offered_products] = 1
        t = tuple(o_set_arr)
        # check if os already encountered
        if t in offersets:
            sales[offersets[t], chosen_product] += 1
        else:
            offersets[t] = len(offersets)
            choice_arr = np.zeros(num_prods)
            choice_arr[chosen_product] = 1

            if len(membership) == 0:
                membership = np.array([o_set_arr])
                sales = np.array([choice_arr])
            else:
                membership = np.vstack((membership, o_set_arr))
                sales = np.vstack((sales, choice_arr))

    return membership, sales


def test_estimator():
    membership_train, sales_train = read_sample_data()
    num_os, num_prods = membership_train.shape
    prods_chosen = sales_train.nonzero()[1]
    prod_sales = sales_train[sales_train > 0]
    n_obs_per_offerset = np.sum(sales_train > 0, 1)
    membership_train_expanded = np.repeat(membership_train, n_obs_per_offerset, axis=0)
    # create Frank Wolfe estimator
    mixEst = FrankWolfeMNLMixEst()
    # change num_iters based on number of classes to be estimated
    mixEst.fit_to_choice_data(prods_chosen, membership_train_expanded, prod_sales, num_iters=10)
    # predict probabilities on out-of-sample transactions
    membership_test, sales_test = read_sample_data(in_sample=False)
    predicted_probs = mixEst.predict_choice_proba(membership_test)[membership_test > 0]
    prod_sales_test = sales_test[membership_test > 0]
    # compute loss on test set
    print('Loss on test set: %.4f' % mixEst.compute_optimization_objective(predicted_probs, prod_sales_test))

if __name__=="__main__":
    test_estimator()