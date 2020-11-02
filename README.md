# Customer-Retention-Predictor

A predictive model which predicts if a customer's bank balance will go below the bank's minimum balance based on his financial and personal status. This model is implented using the python language. The data-set used to train this model, has the following columns/features.


Column Name (Feature)  -	Description

1.customer_id 	       - Customer ID
2.vintage	             - Vintage of the customer with the bank in number of days
3.age                  -	Age of customer
4.gender               -	Gender of customer
5.dependents	         - Number of dependents
6.occupation	         - Occupation of the customer
7.city                 -	City of customer
8.customer_nw_category - Net worth of customer (3:Low 2:Medium 1:High)
9.branch_code	         - Branch Code for customer account
10.days_since_last_transaction -	No of Days Since Last Credit in Last 1 year
11.current_balance	   - Balance as of today
12.previous_month_end_balance	 - End of Month Balance of previous month
13.average_monthly_balance_prevQ	- Average monthly balances (AMB) in Previous Quarter
14.average_monthly_balance_prevQ2	 - Average monthly balances (AMB) in previous to previous quarter
15.current_month_credit - Total Credit Amount current month
16.previous_month_credit -Total Credit Amount previous month
17.current_month_debit	 - Total Debit Amount current month
18.previous_month_debit	- Total Debit Amount previous month	
19.current_month_balance - Average Balance of current month
20.previous_month_balance - Average Balance of previous month
21.churn - Average balance of customer falls below minimum balance in the next quarter (1/0) 1-Yes. 0-No.


I split the original data set into a training data set and a testing data set.
I trained the model using the training set, and I tested the model using the testing data set.

