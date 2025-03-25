## Instructions
```bash
pip install -e .
```

## Run
```bash
cd modeling
inpsect eval initial_generation.py
```


The model that is defined within the environmental variable is the one that is used to conduct the intermediate generation of code after the inital model implementatation.


Currently skips trials where the target value is Nan or None.



##### TODO ###########

- [ ] Dynamically retrieve values of analysis paramters (probably best having user specify)
     - [X] target_value
     - [ ] group to analyze by
     - [ ] particpant identifier
     
- [ ] Add different loss functions in model_utils.py
     - [ ] Predicting numerical values
     - [ ] Predicting categorical values ?

- [ ] Adjust (or ensure) the current negative log likelihood function to work with multiple categories
     - Categorical Cross-Entropy Loss: For multi-class problems (predicting probabilities across multiple possible actions)
     - Formula: -Σ(y_i * log(p_i))


Currently, participant ID is hard coded to be "ID"
[ ] allow user to pass in this value within the configs


- [] BIC is being incorrectly calculated, and likely taking into account trials even if they are not relevant to the model (as seen by difference in BIC between only role 1 dataset and combined dataset for the UG)