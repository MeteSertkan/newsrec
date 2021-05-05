# Welcome to newsrec 👋

In this repository we work on Neural News Recommendation methods.
Currently we re-implemented following models:
- [LSTUR](https://www.aclweb.org/anthology/P19-1033/)
- [NAML](https://arxiv.org/abs/1907.05576)
- [NRMS](https://www.aclweb.org/anthology/D19-1671/)
- [SentiRec](https://www.aclweb.org/anthology/2020.aacl-main.6.pdf) 

## How to train the models
1. Get the MIND dataset from [here](https://msnews.github.io/index.html)
    * We used in our experiment the SMALL version. Inparticular, the training set for validation and training and the validation set as holdout.
2. Install [conda](https://docs.conda.io/en/latest/)
3. We have exported our environment in which we run our expertiments. You can create an environment using the environment.yml as following: ``conda env -f environment.yml``
4. Run the datapreprocessing
    * Use ``project/data/data_preprocess --config project/config/data/mind_recsys2021``
    * Change the source and target directories in the config accordingly
    * If you use another version of MIND datast (e.g., large) change the config accordingl. You will be prompted with infos on which parameters to change
5. Now you can start training
    * Run for example ``project/train --config project/config/model/sentirec/bert_lambda0p2_mu2.yaml``
    * This will start the training of SentiRec in a predefined Setting.
    * Currently all hyperparameters are either tuned onthe created validation set or referr to the values as set in the according papers.
    * You can continue training (if you had to stop for some reasong) as following: ``project/train --config project/config/model/sentirec/bert_lambda0p2_mu2.yaml --resume path_to_ckpt`` , where you have to provide a path to a certain checkpoint

## How to test
1. Train a model :)
2. To test the trained model on the hold out set run: ``project/est --config project/config/model/sentirec/bert_lambda0p2_mu2.yaml --ckpt path_to_ckpt``, where you have to provide a path to the checkpoint you wan tto test
3. Only test the models once, right before you want to report or submit (after you did your tuning etc. )
    
