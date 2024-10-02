## First Documentation of Results with my Pipeline

### Planned steps for this Project

- Here i tried to reproduce the Results from the PeptideBERT Paper. I used the same Data and the same Hyperparameters as in the Paper. 
- I Still need to tweek the MLM train with stuff like Curriculum Learning. There is still lot of optimization to do.

- [x] **Step 1:** Show Results of PeptideBERT Train with normal whitelab data
  - [x] **Step 1.1:** Show ablation Test Results
  - [ ] **Step 1.2:** TODO Cluster with model
- [x] **Step 2:** Show Results of PeptideBERT Train without Data Leakage
- [x] **Step 3:** Show Results of PeptideBERT Train with our Data and w/o Leakage
  - [ ] **Step 3.1:** TODO Cluster with model
- [ ] **Step 4:** Show Results of MLMPeptideBERT and compare to protBERT
  - [ ] **Step 4.1:** Initial Accuracy of mlm task train should be accuracy of protBERT for our peptide Task
  - [ ] **Step 4.2:** Show Results with Curriculum Learning
-[ ] **Step 5:** Retrain binary PeptideBERT on our new MLMBERT
  - [ ] **Step 5.1:** TODO Cluster with model

### Step 1 - 3:
- Due to data leakage and predicted label Problems I couldn't follow the task for reproducing peptideBERTs results.
- Instead, I still try to fix the class imbalance in the dataset and try to produce reasonable results with my pipeline
- The following shows the findings of my analysis of peptideBERTs data leakage and the predicted label problem

### Train run for PeptideBERT on original Algorithm but with my data_leakage_viewer
- This Function searches for leaked data in the train, val and test datasets

![leakage](./pictures/data_leakage_unfiltered.png)

- When allowing leaked data into the train algorithm we get pretty good results

![leakage_results](./pictures/leakage_result.png)

- When filtering out the leaked data we get the following results

![leakage_results_filtered](./pictures/no_leakage_tmp_result.png)

- I could repdroduce PeptideBERTs results with my Pipeline
    


# The Bigger Real Big Problem

- I was curious about the minor impact of the data leakage on the results of the model, so I took a look into the predictions a bit more closely and found the Following:





- I printed out the labels of the Prediction Array each epoch for better tracking
  -  After the first epoch the model already predicts only 0. 

![big problem](./pictures/big_problem.png)

- ### Thats the data label distribution

![distribution](./pictures/data_distribution.png)


- by predicting only 0 it will naturally achieve 80% accuracy since 80% of the data is labeled 0

- Further in Training the bias will get pushed to the initial distribution of the labels of our data, since we're comparing the loss with our ground truth


![epoch problem](./pictures/epoch_big_problem.png)


- it's the same when filtering out leaked data

![epoch no leakage](./pictures/problem_with_out_leaked.png)

- To avoid training with fewer data points I first tried a RandomWeightedSampler by applying label weights

- The Random Sampler kindah seems to fix the predictions in the first epoch, but it jumps wildly
  - May that's due to the class imbalance in the dataset. We sampling train but presenting an imbalanced eval set

![weighted epochs](./pictures/weighted_imbalance.png)

- That's also indicated here sinc eval loss and eval accuracy seems to be pushed to a fixed output but train loss is fluctuating and seems shrinking normaly

![first binary](./pictures/weighted_first.png)

- and a second run with higher early stop params

![second binary](./pictures/weighted_second.png)

- The data always seems to fall to a fixed distribution of labels. This indicates that the model is learning a distribution not a pattern in the sequence

- [ ] Apply RandomSampler also to Eval dataset!!!

- [ ] See if clustering is at least possible with the model

## Proof of Concept
### Step 4 Proof of Concept: MLMPeptideBERT

- mlm task looks good so far. Naive Implementation of MLM Task with PeptideBERT provides stable results so far.
- Plotted is eval_accuracy, i didnt manage to capture train_accuracy so far. It's on TODO

![MLM Task](./pictures/first_mlm.png)
- Loss seems to decrease in a reasonable manner
- Accuracy seems reasonable too, since the model has way more choices for a given mask then in binary classification. If there would be any issues with the predictions it should be more random or with lower accuracy
- [ ] Also plot prediction distribution here. Could be interesting to see if the model is biased towards a certain amino acid
### Step 4.1 
- The accuracy for reconstructing masked bioactive sequences of protBERT should be the initial accuracy of our fine_tuning process

### Step 4.2 Curriculum Learning

- [ ] Implement Curriculum Learning for MLM Task

### Step 5 Proof of Concept: Binary PeptideBERT

- [ ] Still in Progress due to severe bug found in PeptideBERT train algorithm