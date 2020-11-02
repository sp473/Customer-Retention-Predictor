#Importing the packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, precision_score, recall_score, precision_recall_curve
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


#loading and displaying the data
df = pd.read_csv('churn_prediction.csv')
df.head()
#Missing Values
#Before we go on to build the model, we must look for missing values within the dataset as #treating the missing values is a necessary step before we fit a model on the dataset.

pd.isnull(df).sum()

#Replace values ‘male’ and ‘female’ with 1 and 0 respectively and if it’s a missing value we insert -1
dict_gender = {'Male': 1, 'Female':0}
df.replace({'gender': dict_gender}, inplace = True)
df['gender'] = df['gender'].fillna(-1)

#If ‘dependents’ is missing assume it as 0 and for ‘occupation’ we replace the missing value with ‘self employed’
df['dependents'] = df['dependents'].fillna(0)
df['occupation'] = df['occupation'].fillna('self_employed')

#Replacing the missing values of ‘city’ with value 1020
df['city'] = df['city'].fillna(1020)
#Replacing the missing value for the below column with value 999
df['days_since_last_transaction'] = df['days_since_last_transaction'].fillna(999)



# Convert occupation to one hot encoded features. Its because the logistic regression object model , takes in only numbers
df=pd.concat
([df,pd.get_dummies(df['occupation'],prefix=str('occupation'),prefix_sep='_')],axis = 1)

#The columns in the list ‘num_cols’ form the baseline data set which will be used to train the baseline model
num_cols = ['customer_nw_category', 'current_balance',
            'previous_month_end_balance', 'average_monthly_balance_prevQ2', 'average_monthly_balance_prevQ',
            'current_month_credit','previous_month_credit', 'current_month_debit', 
            'previous_month_debit','current_month_balance', 'previous_month_balance']
for i in num_cols:
    df[i] = np.log(df[i] + 17000)

#scaling the values
std = StandardScaler()
scaled = std.fit_transform(df[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)


df_df_og = df.copy()
df = df.drop(columns = num_cols,axis = 1)
df = df.merge(scaled,left_index=True,right_index=True,how = "left")



#dropping irrelevant columns
y_all = df.churn
df = df.drop(['churn','customer_id','occupation'],axis = 1)


#creating the baseline data frame
baseline_cols= 
['current_month_debit','previous_month_debit','current_balance','previous_month_end_balance','vintage','occupation_retired','occupation_salaried','occupation_self_employed', 'occupation_student']
df_baseline = df[baseline_cols]



# Splitting the data into Train and Validation set
xtrain, xtest, ytrain, ytest = train_test_split(df_baseline,y_all,test_size=1/3, random_state=11, stratify = y_all)
model = LogisticRegression()
model.fit(xtrain,ytrain)
pred = model.predict_proba(xtest)[:,1]


#storing the predicted target values to a variable ‘label_pred”
pred_val = model.predict(xtest)
label_preds = pred_val


#creating and plotting a confusion matrix
cm = confusion_matrix(ytest,label_preds)


def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=['Predicted: No','Predicted: Yes'], yticklabels=['Actual: No','Actual: Yes'], cmap=cmap)

plot_confusion_matrix(cm, ['No', 'Yes'])



# Recall Score
recall_score(ytest,pred_val)
precision_score(ytest,pred_val)



#A function that basically takes the data set and the model we built as inputs and perform k fold cross validation and calculate the precision and recall of each fold.
def cv_score(ml_model, rstate = 12, thres = 0.5, cols = df.columns):
    i = 1
    cv_scores = []
    df1 = df.copy()
    df1 = df[cols]
    
    # 5 Fold cross validation stratified on the basis of target
    kf = StratifiedKFold(n_splits=5,random_state=rstate,shuffle=True)
    for df_index,test_index in kf.split(df1,y_all):
        print('\n{} of kfold {}'.format(i,kf.n_splits))
        xtr,xvl = df1.loc[df_index],df1.loc[test_index]
        ytr,yvl = y_all.loc[df_index],y_all.loc[test_index]
            
        # Define model for fitting on the training set for each fold
        model = ml_model
        model.fit(xtr, ytr)
        pred_probs = model.predict_proba(xvl)
        pp = []
         
        # Use threshold to define the classes based on probability values
        for j in pred_probs[:,1]:
            if j>thres:
                pp.append(1)
            else:
                pp.append(0)
         
        # Calculate scores for each fold and print
        pred_val = pp
        roc_score = roc_auc_score(yvl,pred_probs[:,1])
        recall = recall_score(yvl,pred_val)
        precision = precision_score(yvl,pred_val)
        sufix = ""
        msg = ""
        msg += "ROC AUC Score: {}, Recall Score: {:.4f}, Precision Score: {:.4f} ".format(roc_score, recall,precision)
        print("{}".format(msg))
         

# Save scores
        cv_scores.append(roc_score)
        i+=1
    return cv_scores





#to display the scores of the basline model
baseline_scores = cv_score(LogisticRegression(), cols = baseline_cols)

#to display the score of the entire model using the entire data set
all_feat_scores = cv_score(LogisticRegression())


#to plot a bar graph, visualizing the performances of the k folds (parts) of the baseline model and the full model
results_df = pd.DataFrame({'baseline':baseline_scores, 'all_feats': all_feat_scores})
results_df.plot(y=["baseline", "all_feats"], kind="bar")
