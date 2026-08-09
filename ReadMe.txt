Masters thesis source code.
Tiaan Maré


Folders overview:

Utilities - This folder contains all the shared configurations as well as optuna approach. In addition it contains Plots_Metrics which calculates the metrics.

Automation_Scripts - This folder contains the batch runners used to launch experiments and generate analysis outputs. It covers baseline regression and tree runs, feature-selection experiments, and phase-level result analysis scripts.

CodeBase_Experiments - This folder contains the main experimental codebase, from CMAPSS data processing through baseline ML/DL models, feature selection, intra-dataset transfer learning, and DANN / Costa domain-adaptation studies.

NB! the Costa implemantation is based on the following publication:

Da Costa, P. R. D. O., Akçay, A., Zhang, Y., and Kaymak, U. (2020). Remaining useful
lifetime prediction via deep domain adaptation. Reliability Engineering & System Safety,
195:106682

I highly recommend to give the publication a read before implementing the DANN your self.