## First Documentation of Results with my Pipeline

- our model -> The model i fine tuned on startpep data with binary classifier
- peptideBERT model -> Model based on protBERT with binary classifier
- Here shown Plots are not made in chronological order, due to progress in Development. There where no alterations of
  the algorithm but just more ways to fetch and visualize stats and metrics
    - During this process all generated plots seemed reproducable and stable and results

### Planned steps for this Project

- Here i tried to reproduce the Results from the PeptideBERT Paper. I used the same Data and the same Hyperparameters as
  in the Paper.
- I Still need to tweek the MLM train with stuff like Curriculum Learning. There is still lot of optimization to do.

- [x] **Step 1:** Show Results of PeptideBERT Train with normal whitelab data
    - [x] **Step 1.1:** Show ablation Test Results
    - [x] **Step 1.2:** TODO Cluster with model
- [x] **Step 2:** Show Results of PeptideBERT Train without Data Leakage
- [x] **Step 3:** Show Results of PeptideBERT Train with our Data and w/o Leakage
    - [x] **Step 3.1:** TODO Cluster with model
- [x] **Step 4:** Show Results of MLMPeptideBERT and compare to protBERT
    - [x] **Step 4.1:** Initial Accuracy of mlm task train should be accuracy of protBERT for our peptide Task
    - [ ] **Step 4.2:** Show Results with Curriculum Learning ***TODO***
-[x] **Step 5:** Retrain binary PeptideBERT on our new MLMBERT
    - [x] **Step 5.1:** TODO Cluster with model

### Step 1 - 3:

- Due to data leakage I couldn't follow the task for reproducing peptideBERTs results.
- I now try to show the effects of the Data Leakage on the model and the results

### Train run for PeptideBERT on original Algorithm but with my data_leakage_viewer

- This Function searches for leaked data in the train, val and test datasets

![leakage](./pictures/data_leakage_unfiltered.png)

- When allowing leaked data into the train algorithm we get pretty good results

![leakage_results](./pictures/leakage_result.png)

- When filtering out the leaked data we get the following results

![leakage_results_filtered](./pictures/no_leakage_tmp_result.png)

- I could repdroduce PeptideBERTs results with my Pipeline
    - Only difference with my algorithm, I use grad_norm_clipping to avoid exploding gradients

![reproduced](./pictures/reproduced.png)

#### Learning Curves generated through my algorithm

- The Learning Curve for the leaked variant of PeptideBERT
  ![leak_curve](./pictures/leak_curve.png)

- The Learning Curve for the non leaked variant of PeptideBERT
  ![non_leak_curve](./pictures/non_leak_curve.png)

# Is this a Problem?

- I was curious about the minor impact of the data leakage on the results of the model, so I took a look into the
  predictions a bit more closely and found the Following:

- I printed out the labels of the Prediction Array each epoch for better tracking
    - After the first epoch the model already predicts only 0.

![big problem](./pictures/big_problem.png)

- ### Thats the data label distribution

![distribution](./pictures/data_distribution.png)

- by predicting only 0 it will naturally achieve 80% accuracy since 80% of the data is labeled 0

- I plotted the distribution of predicted labels for each batch in training and evaluation
    - Area painted red representing the total of predicted 1 labels
    - Area painted green representing the total of predicted 0 labels

### These are the results for the leakage variant of PeptideBERT

- Label Prediction for Epochs
  ![epoch wise](./pictures/binary_label_abundance.png)
- Label Prediction for Train batches
  ![batch wise](./pictures/train_batch_wise_label_prediction_label_abundance.png)

### These are the results for the non leakage variant of PeptideBERT

- Label Prediction for Epochs
  ![nonleaked epoch wise](./pictures/non_leaked_label.png)
- Label Prediction for Train batches
  ![nonleaked batch wise](./pictures/batch_wise_non_leaked.png)

#### Testing this further

- Anyway, the test I made looked good.
    - I used the Human Atlas Human Peptide Dataset for cleary non bioactive hemolytic Peptides
    - I used the hemolytic positiv data from our dataset for bioactive Peptides
        - The Idea, The human atlas data we should always see 0 since they are not hemolytic nor even bioactive, for our
          data we should see always 1 as prediction

- This plots shows a test plot for the predicted labels for leaked variant of PeptideBERT
    
![label_prediction_bias](./pictures/leak_label_bias_plot.png)

- This plots shows a test plot for the predicted labels for non leaked variant of PeptideBERT
![label_prediction_bias](./pictures/non_leak_label_bias_plot.png)
### Cluster

- Due to the data leakage problem I couldn't find the time to look deeper into the right params for the clustering
  algorithm
    - For tsne is tried different perplexities ranging from 5-50 like stated in the documentary, i also set max_iter to
      3000
        - I couldn't reproduce the tsne results so far. I contacted the author of the paper for help
        - Cluster Graphs shown in the following are first views on the data, this will improve with time

#### Leaked Variant

- t-SNE
  ![tsne_leaked_pep](./pictures/tsne_leaked_pep.png)
- UMAP
  ![umap_leaked_pep](./pictures/umap_leaked_pep.png)
- PCA
  ![pca_leaked_pep](./pictures/pca_leaked_pep.png)

#### Non Leaked

- t-SNE
  ![tsne_nonleaked_pep](./pictures/tsne_nonleaked_pep.png)
- UMAP
  ![umap_nonleaked_pep](./pictures/umap_nonleaked_pep.png)
- PCA
  ![pca_nonleaked_pep](./pictures/pca_nonleaked_pep.png)

## Proof of Concept MLM Task Fine Tuning

### Step 4 Proof of Concept: MLMPeptideBERT

- mlm task looks good so far. Naive Implementation of MLM Task with PeptideBERT provides stable results so far.
- Plotted is eval_accuracy, I didn't manage to capture train_accuracy so far. It's on TODO

![MLM Task](./pictures/first_mlm.png)

- Accuracy raise and loss decrease seems reasonable.
- [ ] Also plot prediction distribution here. Could be interesting to see if the model is biased towards a certain amino
  acid

### Step 4.1

- The accuracy for reconstructing masked bioactive sequences of protBERT should be the initial accuracy of our
  fine_tuning process

### Step 4.2 Curriculum Learning

- [ ] Implement Curriculum Learning for MLM Task

### Step 5 Proof of Concept: Binary PeptideBERT

- I trained a variant with leaked data to compare the results to the original PeptideBERT

#### Leaked Variant

- Learning Curve
  ![our_learning_leak](./pictures/our_learning_leak.png)

- Batch Wise Label Prediction
  ![batch wise](./pictures/ourBERT_batchwise.png)

- Label bias test
  ![label_prediction_bias](./pictures/our_bias_leak.png)

- Epoch wise Label Prediction
  ![epoch wise](./pictures/epoch_label_leak_our.png)

#### Non Leaked Variant

- Learning Curve
  ![our_learning_nonleak](./pictures/our_learning_nonleak.png)
- Batch Wise Label Prediction
- [ ] Did not Compute this here bc this Plot needs ALOT of computation time. But The results should be abstract-able
  from the epoch wise plot

- Epoch wise Label Prediction
  ![epoch wise](./pictures/epoch_label_nonleak_our.png)

- Label bias test
![label_prediction_bias](./pictures/our_label_bias_plot.png)  

### Cluster

##### Leaked Variant:

- t-SNE
  ![tsne_leaked_our](./pictures/tsne_leaked_our.png)
- UMAP
  ![umap_leaked_our](./pictures/umap_leaked_our.png)
- PCA
  ![pca_leaked_our](./pictures/pca_leaked_our.png)

##### Non Leaked Variant:

- t-SNE
  ![tsne_nonleak_our](./pictures/tsne_nonleak_our.png)
- UMAP
  ![umap_nonleak_our](./pictures/umap_nonleak_our.png)
- PCA
  ![pca_nonleak_our](./pictures/pca_nonleak_our.png)

  