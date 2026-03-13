This document explains the purpose of each branch used in the development workflow.

main

This branch contains the baseline model pulled directly from the emg2qwerty project in the calvin pang directory. This includes the standard Readme and similar setup information. Tested on local and Google Collab based environments.

Used as a baseline for development that was augmented by modifying the lighning.py, base, and model files to ensure modular use

annasusanto

Anna Susanto's development branch. Used to develop, train, and test the Transformer model using the base as the baseline

dev/jiustine_colab

Justine Encontro's development branch. Modularized RNN selector that contains lone implementations of lstm, GRU, and RNN architectures in the customized lightning file that are callable by the base.yaml file


dev/shervin

Sgervub Sahidi initial dev branch. Not significant for the project outside of codebase development

dev/shervin-test 

Shervin Shahidi's development branch. Fully developed and matured CNN and BiLSTM model containing lessons learned and modular appraoches that maximized performance. This was the finalized best model.

dev/sina

Sina Ghadimi's development branch. Used for initial dataset to assess and visualize data to improve report findings and assessments
