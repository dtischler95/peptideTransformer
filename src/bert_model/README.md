## Fine Tune Notes

### Implement 
- [x] Implement ReduceLROnPlateau
- [x] Implement Gradient Clipping
- [x] Implement AdamW
  - [ ] Did i do it the right way passing weight decay with init?
- [x] Implement OneCycleLR
- [x] Implement Curriculum Learning
- [x] Find train_accuracy for learning curves

### Notes

- So far excessively tested on mlm algorithm
- mlm with large train data size seems to tend to exploding gradients. saw nans in some eval_losses. But only eval_loss.
  - Seems to appear after huge accuracy increase/decrease

### Ideas

- Generate Auto config generation for training in main
- Implement something to modify the config 