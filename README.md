## Fine Tune ProtBERT for Peptide tasks

### To-Do
#### Data
- [ ] Think of better DataFolder Structure

#### Preprocessing
- [ ] Make data_preprocessing Jupyter Notebook more beautiful
- [ ] Train_Data_viewer.py -> make sense of statistics 
  - [ ] Add data to Train Dataset
- [x] Need datafile with and without filtered ambiguous sequences * Function now accepts bool
- [x] Check for data leakage in Train Data
- [ ] Change signature of get_overall_stats so that dfs are given as argument so i can use it in notebook

#### Training
- [ ] Implement MLM training
- [ ] Train StarPep MLMBert
- [ ] Refactor src Folder
- [ ] Look after max_length parameter
- [ ] Add Jupyter Notebook for Training

#### Evaluation
- [ ] Think of Evaluation Pipeline
- [ ] Change Polars usage in hemo_clustering to pandas
- [ ] Add Evaluation Jupyter Notebook


### Content

- This repository contains the code to fine-tune ProtBERT for peptide tasks. The code is based on the Huggingface Transformers library and the ProtBERT model. This Repository is structured as follows:  
    
    - `data/`: Contains the sequence data
      - `data_for_data_viewer/`: Contains the data for the train_data_viewer.py in `data_analysis/`
      - `data_from_database/`: Contains the data downloaded data from DBs with some basic parsing
      - `sequence_analysis/`: Contains sequences which where sorted out by our data_preprocessing for later analysis
      - `train_data/`: Contains data ready for training
      - `whitelab_data/`: Contains the data used for train [PeptideBERT](https://github.com/ChakradharG/PeptideBERT/tree/masterhttps://github.com/ChakradharG/PeptideBERT/tree/master) 
    - `data_analysis/`: Contains .py scripts for further train data analysis. Includes Cluster Analysis and Data Point comparisons so far
    - `data_preprocessing/`: Contains .py scripts for preprocessing our database and whitelabs data. 
    - `models/`: Contains the trained models
    - `results/`: Contains the results of the evaluation

