# NonparamMixedLogitEstimation
This folder contains code of the nonparameteric estimator for the mixed logit model proposed in the [A conditional gradient approach for nonparametric estimation of mixing distributions](https://pubsonline.informs.org/doi/10.1287/mnsc.2019.3373) paper.

## How to run
`python run_estimator.py`
We use python 3. The file reads the data in the json format given in the example input file ${\tt test_instance.dt}$ and fits a latent class logit model (aka LC-MNL model) on the in-sample transactions provided in the input file.

## Utility model
The version provided here considers the special case when there are no product features, i.e. the utility of customer i in class k for product j is of the form $\beta_{kj} + \epsilon_{ij}$, where $\epsilon_{ij}$ is the error term and $\beta_{kj}$ is the mean utility. 
So the output for a $K$ class model with $n$ products consists of the mixing proportions $q_1, q_2, \ldots, q_K$ and the mean utilities $(\beta_{kj} : \forall k \in [K], j \in [n])$.

## Estimator description
The code for the estimator is in `frank_wolf_lc_mnl.py`. The class variable ${\tt mix_props}$ stores the mixing proportions in a 1-d array and the variable ${\tt coefs_}$ stores the beta parameters in a 2-d array, such that the entry in row $k$ and column $j$ corresponds to $\beta_{kj}$. Other variables are described in the file.

The main method in the estimator is `fit_to_choice_data()`, which takes as input the membership matrix (binary matrix that encodes whether a product is offered in each offerset) and the number of sales for each product in each offerset. We transform the data from the provided input file to this format. See the documentation of the `fit_to_choice_data()` for more details on the input format. The following arguments to the `fit_to_choice_data()` method can be modified based on the application:

1. $\texttt{num_iters}$ : this is the number of iterations to run the estimation for. As mentioned in the paper, this provides an upper bound for the number of latent classes in the estimated LC-MNL model.
2. $\texttt{init_coefs}$ and $\texttt{init_mix_props}$: the initial betas and mixture proportions. 

After estimating the model, you should use the `predict_choice_proba()` function to predict choice probabilities on out-of-sample-transactions, which are also provided in the example instance ${\tt test_instance.dt}$. An example of how to do this is provided in the ${\tt run_estimator.py}$ file.

## Dependencies
The code has been tested with the following (main) dependencies:

numpy==1.20.3

scipy==1.7.1

multiprocess==0.70.12.2

ipython==7.26.0
