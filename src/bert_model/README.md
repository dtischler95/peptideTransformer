## Fine Tune Notes

### Notes

- So far excessively tested on mlm algorithm
- mlm with large train data size seems to tend to exploding gradients. saw nans in some eval_losses. But only eval_loss.
  - Seems to appear after huge accuracy increase/decrease

### Ideas

- Implement something to modify the config 