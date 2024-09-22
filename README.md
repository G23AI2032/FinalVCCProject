**Loan Application Status Prediction Using ML & deploy in GCloud**

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problemStatement)
- [Document Design](#documentDesign)
- [Data Description](#dataDescription)
- [Machine Learning Models](#machineLearningModels)
- [Created Google Collab Notebook](#createdGoogleCollabNotebook)
- [Launched Google CLI](#launchedGoogleCLI)
- [Deployed to Kubernetes](#deployedToKubernetes)
- [Running the application in Google cloud](#runningTheApplicationInGoogleCloud)
- [Sample screens](#sampleScreens)
- [Author](#author)

##**Introduction:**

Loan Application Status Prediction involves the use of predictive models to assess the likelihood of a loan being approved or rejected based on applicant data. By leveraging historical data and applying algorithms to uncover patterns, machine learning can make accurate predictions about the approval status of new loan applications. 
The complete video of project is available on YouTube: Loan status Prediction using Machine Learning and deployed to GCP (youtube.com)
GitHub Link: G23AI2032/FinalVCCProject (github.com)

##**Problem Statement:**

The goal is to classify provided data into status "Approve" and "Reject" categories based on the content.
The challenge is to optimize model performance using ensemble learning methods like Random Forest while identifying the most significant features through feature selection.
And to deploy the same pkl file to Google cloud.


##**Document Design:**

  ![image](https://github.com/user-attachments/assets/99db6cb1-ddc7-439a-99ca-8d16d167c501)

 
##**Data Description:**

The dataset consists of features extracted from form data, possibly from UI Screen, where users post his/her information. The features include:
      ●	Gender (0 for Male, 1 for Female)
      ●	Marital Status (0 for No, 1 for Yes)
      ●	Dependents 
      ●	Education (0 for Graduate, 1 for Not Graduate)
      ●	Self-Employed (0 for No, 1 for Yes)
      ●	Loan Amount 
      ●	Loan Amount Term 
      ●	Credit History (0 for Bad, 1 for Good)
      ●	Property Area (0 for Urban, 1 for Rural)
      ●	Total Income.


##**Machine Learning Models:**

First, we will divide our dataset into two variables X as the features we defined earlier and y as the Loan Status the target value we want to predict.
Models we will use: Random Forest
The Process of Modelling the Data:
•	Importing the model
•	Fitting the model
•	Predicting Loan Status
•	Classification report by Loan Status

![image](https://github.com/user-attachments/assets/c8af6839-e4b3-477c-ba0b-1d42ac8eac35)


##**Created Google Collab Notebook**:

[Uploading{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.experimental import enable_iterative_imputer\n",
    "from sklearn.impute import IterativeImputer\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "train = \"train_u6lujuX_CVtuZ9i.csv\"\n",
    "train = pd.read_csv(train)\n",
    "test = \"test_Y3wMUE5_7gLdaTN.csv\"\n",
    "test = pd.read_csv(test)\n",
    "# Concatenating the train and test data for data preprocessing:\n",
    "data = pd.concat([train,test])\n",
    "# Dropping the unwanted column:\n",
    "data.drop('Loan_ID', inplace=True, axis='columns')\n",
    "# Imputing the missing values:\n",
    "data['Gender'].fillna(data['Gender'].mode()[0], inplace = True)\n",
    "data['Married'].fillna(data['Married'].mode()[0], inplace = True)\n",
    "data['Dependents'].fillna(data['Dependents'].mode()[0], inplace = True)\n",
    "data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace = True)\n",
    "data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace = True)\n",
    "# Next, we will be using Iterative imputer for filling missing values of LoanAmount and Loan_Amount_Term\n",
    "data1 = data.loc[:,['LoanAmount','Loan_Amount_Term']]\n",
    "\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "#Running the imputer with a Random Forest Estimator\n",
    "imp = IterativeImputer(RandomForestRegressor(), max_iter=1000, random_state=0)\n",
    "data1 = pd.DataFrame(imp.fit_transform(data1), columns=data1.columns)\n",
    "\n",
    "data['LoanAmount'] = data1['LoanAmount']\n",
    "data['Loan_Amount_Term'] = data1['Loan_Amount_Term']\n",
    "# So now as we have imputed all the missing values we go on to mapping the categorical variables with the integers.\n",
    "data['Gender'] = data['Gender'].map({'Male':0,'Female':1}).astype(int)\n",
    "data['Married'] = data['Married'].map({'No':0,'Yes':1}).astype(int)\n",
    "data['Education'] = data['Education'].map({'Not Graduate':0,'Graduate':1}).astype(int)\n",
    "data['Self_Employed'] = data['Self_Employed'].map({'No':0,'Yes':1}).astype(int)\n",
    "data['Credit_History'] = data['Credit_History'].astype(int)    \n",
    "data['Property_Area'] = data['Property_Area'].map({'Urban':0,'Rural':1, 'Semiurban':2}).astype(int)\n",
    "data['Dependents'] = data['Dependents'].map({'0':0, '1':1, '2':2, '3+':3})\n",
    "#creating a new feature\n",
    "data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']\n",
    "data.drop(['ApplicantIncome', 'CoapplicantIncome'], axis='columns', inplace=True)\n",
    "new_train = data.iloc[:614]\n",
    "new_test = data.iloc[614:]\n",
    "# Mapping ‘N’ to 0 and ‘Y’ to 1\n",
    "new_train['Loan_Status'] = new_train['Loan_Status'].map({'N':0,'Y':1}).astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating X (input variables) and Y (Target Variable) from the new_train data.\n",
    "x = new_train.drop('Loan_Status', axis='columns')\n",
    "y = new_train['Loan_Status']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Total_Income</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>143.991525</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>5849.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Gender  Married  Dependents  Education  Self_Employed  LoanAmount  \\\n",
       "0       0        0           0          1              0  143.991525   \n",
       "\n",
       "   Loan_Amount_Term  Credit_History  Property_Area  Total_Income  \n",
       "0             360.0               1              0        5849.0  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.head(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Using train test split on the training data for validation\n",
    "X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.827027027027027\n"
     ]
    }
   ],
   "source": [
    "#Building the model using RandomForest\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "rfc = RandomForestClassifier(n_estimators=200)\n",
    "rfc.fit(X_train, y_train)\n",
    "\n",
    "# Getting the accuracy score for Random Forest\n",
    "rfc_pred = rfc.predict(X_test)\n",
    "print(accuracy_score(y_test,rfc_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "with open('rfc.pkl','wb') as file:\n",
    "    pickle.dump(rfc,file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('rfc.pkl','rb') as file:\n",
    "    load_model=pickle.load(file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,\n",
       "       0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0,\n",
       "       1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,\n",
       "       1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,\n",
       "       1, 0, 0, 0, 0, 0, 0, 1, 1])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pred=load_model.predict(X_test)\n",
    "pred"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
 test.ipynb…]()

 
##**Launched Google CLI:**
       step1:
      launch google cli
      
      step2:
      Create a project
      Enable Artifact Registry in gcp
      
      step3:
      Enable cloud run
      
      step4:
      #create a docker repo:
      gcloud artifacts repositories create finalvccproject --repository-format=docker --location=us-west2 --description="Docker repository"
      
      step5:
      #To build docker image:
      gcloud builds submit --region=us-west2 --tag us-west2-docker.pkg.dev/vcc-major/finalvccproject/loanprediction
      
##**Segregated Docker Metrics:**

  ![image](https://github.com/user-attachments/assets/65189304-b200-4381-a138-1f86e369a825)
  ![image](https://github.com/user-attachments/assets/437a8652-79ca-408d-9fbe-207726ebf3ee)
  ![image](https://github.com/user-attachments/assets/9238aa71-92df-4a6b-9354-54b8bdbd4e92)
  ![image](https://github.com/user-attachments/assets/6e277a2c-4c8a-4192-acc3-6132a8480fe7)
  

##**Deployed to Kubernetes:**

     For Scale-up and down we have deployed the same application to Kubernetes.
     On running the application by expose services in Kubernetes,details of the load performance results, if our application is on with more than 50% utilization. Application automatically scales up.
     #Deploy the same application to GKE to enable autoscaling.
     
     ![image](https://github.com/user-attachments/assets/3ae812d7-c2a9-4f39-9b27-eecdb9a8b580)

     ![image](https://github.com/user-attachments/assets/a5977677-bbcc-4d07-b401-62cafd5c7275)
     

      step1: set the region
      gcloud config set compute/region us-west2
      
      Step2: creating k8s cluster in GKE
      gcloud container clusters get-credentials k8s2-cluster 
      
      step3: Deploying docker image to GKE cluster
      kubectl create deployment k8s2 --image=us-west2-docker.pkg.dev/vcc-major/finalvccproject/loanprediction
      kubectl get deployments
      kubectl describe deployment k8s2
      
      step4: Enabling the replicat set
      kubectl scal deployment k8s2 --replicas=5
      
      step6: Enableing Auto scale 
      kubectl autoscale deployment k8s2 --cpu-percent=50 --min=1 --max=5
      
      Step7: Exposing port to external world to access our application.
      kubectl expose deployment k8s2 --name=k8s2-app-service --type=LoadBalancer --port 80 --target-port 5000
      kubectl get service
      
##**Running the application in Google cloud:**

   Deployed in Docker image url: https://loanprediction-437828161516.us-  central1.run.app
   Deployed in Kubernetes url: http://34.94.179.249/
   
##**Sample screens:**

 ![image](https://github.com/user-attachments/assets/89297036-28ec-4c6a-8791-d8dc24b23469)
 ![image](https://github.com/user-attachments/assets/8eb8919a-39c2-4087-8db9-3b24a16536a5)
 ![image](https://github.com/user-attachments/assets/b887a39f-1a4d-4cfb-9453-3b328b768c8d)

##**Usage**

1.Provide the data fields in the form and click on submit button. 2.On click of "Submit", applicant will be prompted with a message "Approved" or "Rejected" upon prediction.

##**Author

G23AI2032**
 
