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

**Introduction:**

Loan Application Status Prediction involves the use of predictive models to assess the likelihood of a loan being approved or rejected based on applicant data. By leveraging historical data and applying algorithms to uncover patterns, machine learning can make accurate predictions about the approval status of new loan applications. 
The complete video of project is available on YouTube: Loan status Prediction using Machine Learning and deployed to GCP (youtube.com)
GitHub Link: G23AI2032/FinalVCCProject (github.com)

**Problem Statement:**

The goal is to classify provided data into status "Approve" and "Reject" categories based on the content.
The challenge is to optimize model performance using ensemble learning methods like Random Forest while identifying the most significant features through feature selection.
And to deploy the same pkl file to Google cloud.


**Document Design:**

  ![image](https://github.com/user-attachments/assets/99db6cb1-ddc7-439a-99ca-8d16d167c501)

 
**Data Description:**

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


**Machine Learning Models:**

First, we will divide our dataset into two variables X as the features we defined earlier and y as the Loan Status the target value we want to predict.
Models we will use: Random Forest
The Process of Modelling the Data:
•	Importing the model
•	Fitting the model
•	Predicting Loan Status
•	Classification report by Loan Status

![image](https://github.com/user-attachments/assets/c8af6839-e4b3-477c-ba0b-1d42ac8eac35)


**Created Google Collab Notebook**:

Created notebook and have comeup with model training .

Generated PKL file from google collab notebook.

 
**Launched Google CLI:**
      
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
      
**Segregated Docker Metrics:**

  ![image](https://github.com/user-attachments/assets/65189304-b200-4381-a138-1f86e369a825)
  ![image](https://github.com/user-attachments/assets/437a8652-79ca-408d-9fbe-207726ebf3ee)
  ![image](https://github.com/user-attachments/assets/9238aa71-92df-4a6b-9354-54b8bdbd4e92)
  ![image](https://github.com/user-attachments/assets/6e277a2c-4c8a-4192-acc3-6132a8480fe7)
  

**Deployed to Kubernetes:**

     For Scale-up and down we have deployed the same application to Kubernetes.
     On running the application by expose services in Kubernetes,details of the load performance results, if our application is on with more than 50% utilization. Application automatically scales up.

         
     ![image](https://github.com/user-attachments/assets/3ae812d7-c2a9-4f39-9b27-eecdb9a8b580)

     ![image](https://github.com/user-attachments/assets/a5977677-bbcc-4d07-b401-62cafd5c7275)
     
 **Deploy the same application to GKE to enable autoscaling.**
 
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
      
**Running the application in Google cloud:**

   Deployed in Docker image url: https://loanprediction-437828161516.us-  central1.run.app
   
   Deployed in Kubernetes url: http://34.94.179.249/
   
**Sample screens:**

 ![image](https://github.com/user-attachments/assets/89297036-28ec-4c6a-8791-d8dc24b23469)
 ![image](https://github.com/user-attachments/assets/8eb8919a-39c2-4087-8db9-3b24a16536a5)
 ![image](https://github.com/user-attachments/assets/b887a39f-1a4d-4cfb-9453-3b328b768c8d)

**Usage**

1.Provide the data fields in the form and click on submit button. 

2.On click of "Submit", applicant will be prompted with a message "Approved" or "Rejected" upon prediction.

**Author

G23AI2032**
 
