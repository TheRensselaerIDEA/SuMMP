import sys
sys.path.append('./ExplainableAI')
import numpy as np
import pandas as pd
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from explainableAI.models import SupervisedBMM, SupervisedGMM
from explainableAI.metrics import calc_metrics, CalculateSoftLogReg, optimalTau,metrics_cluster,sgmmResults
from explainableAI.models.mlModels import *
from explainableAI.metrics.utility import entropy,asymm_entropy
from explainableAI.metrics.ftest_logodds import ftest_uncorr, restest
from explainableAI.visual.clustmap import plotclustmap
from explainableAI.visual import PDF
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
#from clustmap_newborn import plotclustmap
from sklearn.linear_model import LogisticRegression

print("Loading Data...")
sparcs = pd.read_csv("Data/sparcs25%Newborn_DeHos_Outflow_Region.csv") 
print("Data Loaded...")

print("Preparing the data")
d_newborn_tr, d_newborn_te = train_test_split(sparcs, test_size=0.2, random_state = 1512)
Xtrain, Xtest = d_newborn_tr.iloc[:,0:-1].values, d_newborn_te.iloc[:,0:-1].values
ytrain, ytest = d_newborn_tr.iloc[:,-1].values.astype(int), d_newborn_te.iloc[:,-1].values.astype(int)

print("Set Seed")
np.random.seed( seed = 71730 )

# train SBMM model with Log Regression
max_iter = 30
max_iter2 = 30
n_clusters = 7

print("Starting Model...")
modelB = SupervisedBMM( max_iter =max_iter, n_clusters = n_clusters, max_iter2 = max_iter2, verbose = 0, solver="liblinear")
modelB = modelB.fitB( Xtrain = Xtrain, Xtest = Xtest, ytrain = ytrain)

print("Model done, saving...")
modelB.save("SBMM_niter1_30_niter2_30_nclust_7_seed_71730.pkl")
print("Model saved...")