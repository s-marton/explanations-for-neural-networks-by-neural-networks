# Explanations for Neural Networks by Neural Networks
Official implementation of the paper "Explanations for Neural Networks by Neural Networks" by Sascha Marton, Stefan Lüdtke and Christian Bartelt

To replicate the results from the paper, just replace the relevant parameters values in the config of each notebook by the values from the paper and leave all additional parameters unchanged. Please note, that the notebooks 01, 02 and 03 need to be run in subsequent order, while keeping the relevant parameters equal. The used libraries and versiona are contained in the requirements.txt.

The relevant parameters include:
Parameter     | Exemplary Value   | Explanation
------------- | ------------- | -------------
d | 3 | degree
n  | 5  | number of variables
sample_sparsity  | 5  | sparsity of the polynomial (has to be smaller than the maximum number of monomials for variable-degree combination)
polynomial_data_size  | 50,000  | number of functions to generate for lambda-net training
lambda_nets_total  | 50,000  | number of lambda-nets trained (lambda_nets_total <= polynomial_data_size)
lambda_dataset_size | 5,000  | number of samples per polynomial
lambda_dataset_size | 50,000  | number of trained lambda-nets used for the training of I-Net (interpretation_dataset_size <= lambda_nets_total))
noise | 0  | noise level
interpretation_net_output_monomials | 5 | max number of monomials contained in the polynomial predicted by the I-Net (usually equals the sample_sparsity)
