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

