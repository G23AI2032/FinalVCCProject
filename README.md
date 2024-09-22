**Loan Application Status Prediction Using ML & deploy in GCloud**
**1.	Introduction:**
Loan Application Status Prediction involves the use of predictive models to assess the likelihood of a loan being approved or rejected based on applicant data. By leveraging historical data and applying algorithms to uncover patterns, machine learning can make accurate predictions about the approval status of new loan applications. 
The complete video of project is available on YouTube: Loan status Prediction using Machine Learning and deployed to GCP (youtube.com)
GitHub Link: G23AI2032/FinalVCCProject (github.com)

**2.	Problem Statement:**
The goal is to classify provided data into status "Approve" and "Reject" categories based on the content.
The challenge is to optimize model performance using ensemble learning methods like Random Forest while identifying the most significant features through feature selection.
And to deploy the same pkl file to Google cloud.

**3.	Document Design:**
  ![image](https://github.com/user-attachments/assets/99db6cb1-ddc7-439a-99ca-8d16d167c501)

 
**4.	Data Description:**
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

**5.	 Machine Learning Models:**
First, we will divide our dataset into two variables X as the features we defined earlier and y as the Loan Status the target value we want to predict.
Models we will use: Random Forest
The Process of Modelling the Data:
•	Importing the model
•	Fitting the model
•	Predicting Loan Status
•	Classification report by Loan Status
![image](https://github.com/user-attachments/assets/c8af6839-e4b3-477c-ba0b-1d42ac8eac35)

 
**6.	Created Loan Application Prediction UI:**
 ![image](https://github.com/user-attachments/assets/67e6942d-8801-4c3f-bb60-c9fb3761203f)

**7.	Approved Status UI:**
 ![image](https://github.com/user-attachments/assets/31ee3546-1eb1-4761-a16e-cea5e6bc8655)


**8.	Rejected Status UI:**
![image](https://github.com/user-attachments/assets/f738f4b0-7742-4198-a998-d526f79c04f1)

**9.	Created Google Collab Notebook**:
[Uploading test.ipynb…]()
**10.	Generated PKL from the google collab notebook:**
**11. Launched Google CLI:**
**12. Segregated Docker Metrics:**
**13.Deployed to Kubernetes:**
     For Scale-up and down we have deployed the same application to Kubernetes.
     On running the application by expose services in Kubernetes,details of the load performance results, if our application is on with more than 50% utilization. Application automatically scales up.
**14.Running the application in Google cloud:**
**Usage**
1.Provide the data fields in the form and click on submit button. 2.On click of "Submit", applicant will be prompted with a message "Approved" or "Rejected" upon prediction.

**Author
G23AI2032**
 
