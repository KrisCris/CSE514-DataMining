# Description

This is the repo for 2022 SP WUSTL CSE514 Programming Assignments



# HOW TO RUN?

1. Install Python 3.9.10
1. Open Terminal (Or Windows PowerShell)
2. Initialize venv by typing `python3 -m venv venv` (See Official Document)
3. Activate the python venv (See Official Document)
4. Install dependencies by run `pip install -r requirements.txt`
5. Run the python file you want.



# Assignment 1

- It is inside folder `Regression`, so you want to `cd Regression` first.
- Run `regression.py` by `python regression.py` to see the result (You need activate the env first, if it is activated, `(venv)` would show up on the very left of your console prompts).
- Examples of function calls are shown under `if __name__ == '__main__: '`, you should get all result (some of the diagrams) used in the report by running the given code.

- Example of results:

  - ```
    ######## uni-variate linear regression ########
    ######## Col: 0 ########
    #### STD: True ####
    [TRAINING]
    Params: [34.41549554 21.55869624]
    Loss: 228.33110436125853
    Final L2Derivatives: 9.789024272106074e-07
    Steps: 1500
    Training Time: 0s
    R Square on Training Set: 0.22825858392552245
    
    [TESTING]
    LOSS: 81.13880521108854
    R Square: 0.4354355804392598
    
    #### STD: False ####
    [TRAINING]
    Params: [ 0.08195801 12.43353753]
    Loss: 228.47257770672118
    Final L2Derivatives: 0.12738664244089923
    Steps: 1000000
    Training Time: 22s
    R Square on Training Set: 0.2277804149950582
    
    [TESTING]
    LOSS: 81.2192004197624
    R Square: 0.434876189970022
    ```

  - Params are vectors of parameters for your trained model, i.e., [m1, m2, m3, ... , b].

  - Loss is the MSE
  - Steps are the number of iterations in training
  - STD: True meaning the model is trained by standardized dataset; vise versa.