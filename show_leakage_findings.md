## This Notebook is used to show the effect of the leakage of the datasets on the performance of the model


### Train run for PeptideBERT on original Algorithm but with my data_leakage_viewer
- This Function searches for leaked data in the train, val and test datasets

![leakage](./pictures/data_leakage_unfiltered.png)

- When allowing leaked data into the train algorithm we get pretty good results

![leakage_results](./pictures/leakage_result.png)

- When filtering out the leaked data we get the following results

![leakage_results_filtered](./pictures/no_leakage_tmp_result.png)

- I could repdroduce PeptideBERTs results with my Pipeline
    


# The Bigger Real Big Problem



- Thats the data label distribution

![distribution](./pictures/data_distribution.png)




- Prediction only 0 label in the first validation round after an epoch shows a clear sign of overfitting

- ![big problem](./pictures/big_problem.png)

- by predicting only 0 it will naturally achieve 80% accuracy

![epoch problem](./pictures/epoch_big_problem.png)

- Further in Training the bias will get pushed to the initial distribution of the labels of our data, since we're comparing the loss with our ground truth

- it's the same when filtering out leaked data

![epoch no leakage](./pictures/problem_with_out_leaked.png)

- I Tried a RandomWeightedSampler by applying label weights

![first binary](./pictures/weighted_first.png)

- and a second run with higher early stop params

![second binary](./pictures/weighted_second.png)

- The data always seems to fall to a fixed distribution of labels. This indicates that the model is learning a distribution not a pattern in the sequence