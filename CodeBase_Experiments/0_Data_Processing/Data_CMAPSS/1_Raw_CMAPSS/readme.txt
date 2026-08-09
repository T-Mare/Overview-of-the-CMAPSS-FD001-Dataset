
C-MAPSS, or Commercial Modular Aero-Propulsion System Simulation, is a software simulation tool 
that models the behaviour of a large commercial turbofan engine with a control system that closely 
mirrors actual engine operations. NASA’s Prognostics Center of Excellence 
have used this simulation environment to create a set of datasets that are evidently well-established 
benchmark for PdM related research. 
The set of datasets are named FD001, FD002, FD003, and FD004 where they all represent different engine deration 
scenarios (see bellow for their distinction). The datasets are distinguished by the number engine units, 
operating conditions, and the specific fault modes that were simulated.

For more information refer to the Attatched PDF: Damage Propagation Modeling
Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation�, in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

Data Set: FD001
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: ONE (HPC Degradation)

Data Set: FD002
Train trjectories: 260
Test trajectories: 259
Conditions: SIX 
Fault Modes: ONE (HPC Degradation)

Data Set: FD003
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: TWO (HPC Degradation, Fan Degradation)

Data Set: FD004
Train trjectories: 248
Test trajectories: 249
Conditions: SIX 
Fault Modes: TWO (HPC Degradation, Fan Degradation)


The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:
1)	unit number
2)	time, in cycles
3)	operational setting 1
4)	operational setting 2
5)	operational setting 3
6)	sensor measurement  1
7)	sensor measurement  2
...
26)	sensor measurement  26


Data Availability: 
https://www.kaggle.com/datasets/behrad3d/nasa-cmaps [From which retrieved]
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/  [Alternative]


