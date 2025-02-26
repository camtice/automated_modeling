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

##### TODO ###########

- [ ] Go through the codebase and ensure that no hardcoded values are used.
     - [ ] In def negative_log_likelihood() change `actual_decision = trial["accept"]` to `actual_decision = trial["target_value"]` where "target_value" is defined in the configs as the variable the model should be predicting.
     - [ ] Check if group variable is hard coded in `get_bic_summary()`
     