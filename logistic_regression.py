import pandas as pd 
import statsmodels.api as sm 
import math
loansData = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%'))/100, 4))
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))
loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: int(x[:3]))
loansData['IR_TF'] = loansData['Interest.Rate'].map(lambda x: 1 if x >.12 else 0)
intercept = [1] * len(loansData)
loansData['Intercept'] = intercept
ind_vars = ['Intercept', 'Amount.Requested', 'FICO.Score']
df = loansData
logit = sm.Logit(df['IR_TF'], df[ind_vars])
result = logit.fit()
coeff = result.params
print(coeff)
def logistic_function(FicoScore, LoanAmount):
prob = 1/(1 + math.exp(coeff[0] + coeff[2]*FicoScore+coeff[1]*LoanAmount))
if prob > 0.7:
p = 1
else:
p=0
return prob, p
logistic_function(720, 10000)
print("The probability value is above 0.70 so we predict that we do obtain the loan")
