Project Overview
This project will include a machine learning fraud detection model for medical claims using the Medicare Provider Utilization and Payment Data (MPPD), specifically the "Physician and Other Supplier" data (MPOP PS and MPOP P). This data, available from the Centers for Medicare & Medicaid Services (CMS), provides detailed information on provider utilization, payments, and charges, and can be linked with information from the Office of Inspector General's (OIG) List of Excluded Individuals/Entities (LEIE) to identify fraudulent providers.


Research question: Can machine learning models (specifically an unsupervised One-Class SVM) detect anomalous Medicare provider behavior indicative of fraud using utilization and payment data?
Objective: Identify unusual provider behavior patterns without relying on extensive labeled data.
Project Goals
Purpose
The purpose of this project is to develop an advanced fraud detection framework for Medicare provider claims by applying unsupervised machine learning techniques. Fraudulent billing practices cost the U.S. healthcare system billions annually, leading to financial losses and reduced care quality for beneficiaries. Current fraud detection approaches often rely on manually identified patterns or post-hoc investigations, which may fail to detect novel or evolving fraud schemes. This project aims to contribute to the healthcare analytics field by exploring the use of unsupervised learning to proactively identify anomalous billing behaviors that may indicate fraud.
Project Focus
This research focuses on the application of an One-Class Support Vector Machine (SVM) and an Isolation Forest Model to Medicare Provider Utilization and Payment Data (MPOP) to detect anomalous providers. The primary research question is: Can machine learning models accurately predict Medicare provider fraud using utilization and payment data? Key areas of investigation include: identifying patterns of abnormal provider behavior based on claims, payments, and utilization metrics, determining the feasibility of using an unsupervised approach when labeled fraud cases are limited, and validating detection anomalies against the Office of Inspector General’s (OIG) List of Excluded Individuals/Entities (LEIE). 
	Specific Goals
Specific goals include:
Collect and preprocess Medical Provider Utilization and Payment (MPOP) data and integrate it with the OIG LEIE dataset.
Engineer key features that capture financial, utilization, and behavioral characteristics of providers.
Implement a One-Class SVM to model normal provider behavior and flag potential anomalies. 
Evaluate the model’s ability to identify known fraudulent providers using precision, recall, and anomaly score analysis. 
Provide insights into high-risk provider patterns that may inform future CMS/OIG fraud prevention strategies. 
	Expected Outcomes
Primary outcome: a machine learning model capable of flagging anomalous provider behaviors suggestive of fraud. 

Tangible deliverables:
A preprocessed and feature-engineered dataset linking MPOP and LEIE data.
An operational One-Class SVM pipeline for fraud detection.
An evaluation report with metrics, anomaly rankings, and key findings. 

Intangible outcomes:
Insights into fraud-related patterns in Medicare provider billing.
A demonstration of how unsupervised anomaly detection can complement existing fraud prevention efforts. 
Project Description
Project Objective and Scope
The objective of this project is to develop an unsupervised machine learning model capable of identifying anomalous Medicare provider behaviors that may indicate fraudulent activity. The chosen problem domain, healthcare fraud detection, is of significant public importance, as fraudulent claims contribute to rising healthcare costs and misuse of federal funds. By applying a One-Class Support Vector Machine (SVM) to Medicare Provider Utilization and Payment (MPOP) data, this project aims to detect providers exhibiting outlier patterns in billing and utilization. The scope includes data acquisition, preprocessing, model training, and evaluation using a real-world dataset, with validation based on known fraudulent providers from the Office of Inspector General’s (OIG) List of Excluded Individuals/Entities (LEIE).
Data Description
The primary dataset will be the Medicare Provider Utilization and Payment Data (Physician and Other Supplier), also known as the Public Use File (PUF), obtained from the Centers for Medicare & Medicaid Services (CMS). This dataset includes detailed information on provider charges, payments, services, and beneficiaries. The data is organized by National Provider Identifier (NPI) and Healthcare Common Procedure Coding System (HCPCS) code to show how much Medicare pays, what is allowed, and what is billed. To assess model performance, this data will be cross-referenced with the OIG LEIE dataset, version updated on July 7, 2025, which lists providers excluded from federal health programs due to fraudulent or abusive practices. Individuals and entities who have been reinstated are not included in the updated data file. These sources are publicly available, comprehensive, and directly relevant to the problem domain.
Exploratory Data Analysis
Initial exploration will include:
Statistical summaries (mean, median, standard deviation) of key financial and utilization metrics.
Distribution analysis of charges, payments, and service volumes.
Correlation analysis between provider attributes and payment behaviors.
Visualization of outliers using confusion matrices and scatter plots to identify potential anomalies.
The goal is to gain insight into normal provider behavior and inform feature selection for the anomaly detection model.
Data Preparation and Cleaning
Data preprocessing will include:
Removal of duplicate or invalid provider entries.
Handling of missing values using imputation techniques, such as inputting the mean value in null values where available.  where missing values are estimated by modeling each incomplete feature as a function of the other features. This approach preserves relationships between variables and provides more accurate estimates than simple mean imputation.
For outliers, calculate the IQR and flag values below Q1 and above Q3 for numerical values. For categorical values, group uncommon variables as “other” rather than removal, as removal may alter target values.
Normalization and scaling of continuous variables to standardize input ranges for SVM.
Implement SimpleImputer to manage large scale data.
Encoding categorical variables such as provider specialty and geographic location using one-hot encoding.
Feature engineering to create indicators such as average payment-to-charge ratio, number of unique beneficiaries, and total services rendered.
Model Training
The primary model will be a One-Class Support Vector Machine (SVM) with an RBF kernel, trained exclusively on non-LEIE providers to learn the distribution of legitimate billing behavior. Key parameters to be tuned include nu (ν), which controls the proportion of expected outliers, and gamma (γ), which determines the decision boundary’s sensitivity. Training will involve iterative parameter tuning using grid search and cross-validation techniques, where possible, on subsets of the data.
Model Evaluation
Evaluation will be performed by comparing detected anomalies with known fraudulent providers from the OIG LEIE dataset. Key metrics will include:
Precision, recall, and F1-score to measure the model’s fraud identification performance.
Receiver Operating Characteristic Curve and Area Under the Curve (ROC-AUC) to evaluate ranking effectiveness.
Analysis of the top-ranked anomalies to assess plausibility and false-positive trade-offs.
Findings will be reported through visualizations, confusion matrices, and anomaly score distributions.
User-Interface Integration
The project will include a simple user interface to visualize model outputs and facilitate interpretation by stakeholders. All programming of the machine learning model will be conducted in Jupyter Notebooks. Using the “pickle” library to export, the model will be accessible through Flask. Instead of using AWS, Google Cloud, Microsoft Azure, or Oracle Cloud will be integrated for model accessibility. Actions included in the user interface are:
Filtering and ranking capabilities to prioritize providers for investigation.
Exportable reports summarizing the findings for compliance teams or policymakers.

Capstone Complexity
This project demonstrates master’s-level complexity through the integration of advanced machine learning techniques, large-scale healthcare datasets, and comprehensive analytical workflows. Fraud detection in Medicare is a high-impact area with direct financial and policy implications. This project advances the field by demonstrating how unsupervised learning can complement existing rule-based or supervised approaches, particularly in contexts where labeled fraud cases are scarce.
1. Data Complexity
The project utilizes high-dimensional, real-world Medicare Provider Utilization and Payment (MPOP) data, which includes numerous financial and utilization variables across a large population of providers.
Integration with the Office of Inspector General’s (OIG) List of Excluded Individuals/Entities (LEIE) adds a secondary data source for validation, requiring entity resolution and reconciliation of disparate formats.
Extensive preprocessing will be necessary, including missing value handling, feature scaling, and encoding categorical variables, to ensure data quality and compatibility with the machine learning pipeline.
2. Technical Modeling Complexity 
The project employs an unsupervised One-Class SVM algorithm, a non-trivial anomaly detection technique that requires:
Careful parameter tuning (e.g., nu, gamma) to balance false positives and detection sensitivity.
Feature engineering to extract meaningful patterns from heterogeneous healthcare billing data.
Experimentation with dimensionality reduction methods (e.g., PCA) to enhance model stability and interpretability.
The model will be validated using imbalanced, real-world fraud detection metrics, such as Precision-Recall and ROC-AUC, requiring thoughtful evaluation beyond simple accuracy.
3. Statistical and Analytical Depth
Exploratory Data Analysis will include distribution analysis, correlation matrices, and anomaly visualization techniques to understand underlying patterns and detect natural clusters of provider behaviors.
Post-model analysis will explore the characteristics of flagged anomalies and their alignment with known fraudulent patterns, potentially revealing systemic fraud risks or geographic/specialty hotspots.
4. User Interface Interpretability
Anomaly scores generated by the One-Class SVM model will be ranked to highlight high-risk providers for further review.
Compliance teams and policymakers will be able to generate and download summary reports (e.g., CSV or PDF) containing flagged providers, key metrics, and explanatory features.
Software
This project will leverage a combination of development, data analysis, machine learning, and deployment tools to facilitate the full lifecycle of the Medicare fraud detection model, from data preprocessing to model deployment and user interface development.
1. Python (Jupyter Notebook, Visual Studio Code)
Primary Function: Main programming language for data manipulation, analysis, machine learning, and backend development.
Role in Project: Used for exploratory data analysis, feature engineering, training the One-Class SVM model, and integrating with Flask for deployment.
Key Libraries:
pandas, numpy – Data manipulation and analysis.
scikit-learn – One-Class SVM, preprocessing, OneHotEncoder for categorical variables, and evaluation metrics.
matplotlib, seaborn– Data visualization and anomaly score distributions.
pickle – Model serialization for deployment.
2. Flask
Primary Function: Lightweight web framework for deploying the machine learning model.
Role in Project: Hosts the One-Class SVM model, providing API endpoints for scoring new provider data and serving the user interface.
3. User Interface (UI) (Voila) Cloud Platform (Google Cloud, Microsoft Azure, or Oracle Cloud)
Primary Function: ACloud hosting and accessibility for the deployed model and web application.
Role in Project: Ensures scalability, remote accessibility, and integration with stakeholder systems without relying on local infrastructure.
Project Completion Plan
Week 1
Project Setup and Data Acquisition
Define final research objectives and scope.
Obtain Medicare Provider Utilization and Payment (MPOP) datasets and OIG LEIE data.
Set up development environment (Python, VS Code, Jupyter Notebook, Flask) and cloud access.
Week 2
 Exploratory Data Analysis
Conduct statistical summaries of key variables (charges, payments, service volumes).
Visualize distributions, correlations, and preliminary anomalies.
Identify missing values, outliers, and data inconsistencies.
Data Cleaning and Feature Engineering
Handle missing values and remove duplicates or invalid entries.
Scale and normalize continuous variables.
Encode categorical variables using OneHotEncoder (e.g. specialty, state).
Engineer features for fraud detection (average payment-to-charge ratio, total services, unique beneficiaries).
Week 3
Model Development
Implement One-Class SVM for anomaly detection.
Tune hyperparameters (nu, gamma) to balance sensitivity and false positives.
Train model on non-LEIE providers 
Week 4
Model Evaluation and Validation
Evaluate anomaly detection performance using Precision, Recall, F1-score, ROC-AUC.
Compare top-ranked anomalies with LEIE dataset for validation.
Analyze contributing features for interpretability.
Week 5
	User Interface Development
Develop Flask-based UI for visualizing model outputs.
Implement filtering and ranking of flagged providers.
Enable exportable reports (CSV or PDF) for compliance teams or policymakers.
Deploy model and UI on chosen cloud platform (Google Cloud, Azure, or Oracle Cloud).
Week 6
Presentation and Reporting
Compile visualizations, evaluation metrics, and anomaly summaries.
Complete capstone script, including methodology, results, discussion, and UI overview.
Prepare presentation materials and finalize project submission.
Week 7
Finalization
Additional presentation material development.
Record presentation.
Submit all materials. 
Presentation Plan
The final project presentation will be a video, 30 minutes or less, designed to clearly and effectively communicate the objectives, methodology, and outcomes of the Medicare fraud detection project. The presentation will integrate a combination of PowerPoint slides, code walkthroughs, visualizations, and a user interface demonstration to guide viewers through the project workflow from start to finish.

Structure:
Introduction (3–5 minutes)
Overview of Medicare fraud, its impact, and motivation for the project.
Statement of research question: “Can machine learning models accurately predict Medicare provider fraud using utilization and payment data?”
Brief explanation of datasets (MPOP & LEIE) and project objectives.
Data and Methodology (5–7 minutes)
Walkthrough of data acquisition, cleaning, and feature engineering.
Demonstrate handling of missing values, scaling, and categorical encoding (OneHotEncoder).
Description of the One-Class SVM model, hyperparameters, and rationale for using an unsupervised approach.
Brief exploratory data analysis visualizations to illustrate patterns in provider behavior.
Model Training and Evaluation (7–8 minutes)
Live walkthrough of training the One-Class SVM on the dataset.
Presentation of evaluation metrics: Precision, Recall, F1-score, ROC-AUC.
Visualizations of anomaly score distributions, top-ranked anomalies, and feature importance for interpretability.
User Interface Demonstration (5–7 minutes)
Showcase the Flask-based UI hosted on the cloud platform.
Demonstrate filtering, ranking, and report export features.
Highlight how stakeholders can interact with the model to prioritize providers for investigation.
Discussion and Conclusion (3–5 minutes)
Summary of key findings and insights from the model.
Limitations of the study (e.g. unsupervised assumptions, false positives).
Potential future work and applications of the system for fraud prevention.
PowerPoint slides will structure the presentation and highlight key visuals. Live walkthrough of Jupyter Notebook code and Flask interface. Dashboard screenshots and live interaction demonstrating anomaly detection outputs. Graphs and visualizations will clearly communicate results. This combined approach ensures that the presentation guides viewers through the thought process, methodology, and analytical workflow. It clearly shows both technical and practical outcomes of the project. It also provides visual and interactive demonstrations, making the complex process of fraud detection accessible to technical and non-technical stakeholders.
Resources
Centers for Medicare & Medicaid Services Data. (n.d.). Data.cms.gov. https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider
LEIE Downloadable Databases | Office of Inspector General | U.S. Department of Health and Human Services. (n.d.). Oig.hhs.gov. https://oig.hhs.gov/exclusions/exclusions_list.asp
Centers for Medicare & Medicaid Services Data. (2025). Cms.gov. https://data.cms.gov/resources/medicare-physician-other-practitioners-by-provider-data-dictionary
Centers for Medicare & Medicaid Services Data. (n.d.). Data.cms.gov. https://data.cms.gov/resources/medicare-physician-other-practitioners-methodology
