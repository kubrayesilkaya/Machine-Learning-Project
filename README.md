# Machine-Learning-Project
### Predicting Length of Hospital Stay of Patiens
This project aims to predict the length of hospital stay of patients using machine learning. The length of stay of patients in the hospital is evaluated by various parameters such as hospital health services facilities, the number of MRI and tomography devices in the hospital and the number of beds. The hospital stay of patients is estimated with machine learning algorithms using parameters and this prediction information is aimed to be used for various benefits.
The project of estimating the length of hospital stay of patients has the goal of making vital improvements in healthcare systems. It will be vitally beneficial for patients if resource management in healthcare systems responds to needs and requirements with high-accuracy predictions. The project aims to anticipate advancements in healthcare system planning and hospital management. This study, which will ensure that patients receive more efficient health services, will be beneficial for both patients and hospitals.
In briefly, the project of estimating the duration of hospital stay of patients with machine learning using various parameters aims to improve hospital management in terms of planning and resource management, determination of the required number of beds and increasing the quality of healthcare systems.

### Technologies and Tools Used in the Project
The programming language used when developing the project is Python programming language. Python programming language was preferred due to its understandable syntax structure, large number of libraries in the field of artificial intelligence and data science, ready-made functions and wide developer support. With the advantages of the Python programming language, project development stages become more efficient and the chance of focusing on the machine learning project to have a high accuracy rate increases. The Python libraries used in the project are pandas, numpy and scikit-learn libraries.

### Data Analysis and Data Preprocessing
Following the project problem statement, data sets that would be suitable for the project subject were searched. In the project, data set research to be used in machine learning models was conducted on the Kaggle platform. The 'Healthcare Investments and Length of Hospital Stay' data set was selected by analyzing its suitability for the requirements of the project.

After selecting the data set that suits the project requirements, studies were carried out on the selected data set. The columns and information contained in the selected data set were examined in detail with certain functions. Whether there was missing or duplicate data in the data set was checked with certain functions. Various relationships between parameters in the data set were analyzed and important inferences were made. It was observed that there was a linear correlation between the number of bed parameters in the data set and the length of hospital stay parameters. This determined inference shows how important the project is to make improvements in health systems.

Following the data set analysis, the data types in the data set were examined and categorical variables were encoded. One-Hot encoding was performed for the location column in the data set. After data pre-processing, data splitting operations were studied to establish machine learning models. 

### Splitting the Data
When working with machine learning models, the concepts of dependent variable and independent variable are encountered. The dependent variable (y) is our main target variable that we are trying to predict, and the independent variable (x) is the variables we use to predict the dependent variable.

When working with machine learning models, the data set is divided into training data set and test data set. While the training set is used to train the model, the performance of the established model is tested with the test data set. In this project, 80 percent of the data set was reserved for the training set, and the remaining 20 percent of the data set was kept as the test data set.  

![data split](https://github.com/kubrayesilkaya/Machine-Learning-Project/assets/93487264/2048fa29-1012-4142-b60f-a8443ef165ab)

### Model Trainig
When establishing Machine Learning models, the relationship between the dependent variable and the independent variable is modeled. In this project, various regression models that will work with high performance and suit the project data and requirements have been selected. The models established and trained in the project are Linear Regression, K-Nearest Neighbors, Support Vector Machine (RBF Kernel), Decision Tree, Random Forest, Gradient Boosting models.
The figure below includes the models used in this project.

![train](https://github.com/kubrayesilkaya/Machine-Learning-Project/assets/93487264/678c6700-7809-4cc3-bf3b-810d25990d5f)

### Model Evaluation
After machine learning models are established, various model success evaluation methods are applied to evaluate how successful the predictions of this model are. In this project, Mean Squared Error (MSE), Root Mean Squared Error (RMS), Mean Absolute Error (MAE) and R-Squared Evaluation metric methods were used, and model successes were evaluated. The performance of machine learning predictions realized as a result of training the models established with selected machine learning was evaluated.
The application of the selected machine learning model evaluation criteria and the results obtained by applying these criteria are listed below.

![MSE mean squared error](https://github.com/kubrayesilkaya/Machine-Learning-Project/assets/93487264/cc109068-2fdf-4af0-8606-0ae359255602)

![RMSE](https://github.com/kubrayesilkaya/Machine-Learning-Project/assets/93487264/91d8f16f-8f93-48c0-99e6-079374db9052)

![MAE mean absolute error](https://github.com/kubrayesilkaya/Machine-Learning-Project/assets/93487264/3a2d739e-e2c1-41a6-a3a5-541530fc0826)

![r squared](https://github.com/kubrayesilkaya/Machine-Learning-Project/assets/93487264/0f37f326-975b-415e-af02-762ccfdd9fbf)

### RESULTS
After selecting a suitable data set for the project of estimating the length of hospital stay of patients, models suitable for the project requirements were determined. After establishing and training models with the selected model types, the performance of these models was measured by applying various model evaluation criteria. The lower error value calculated for the Mean Squared Error (MSE), Root Mean Squared Error (RMS) and Mean Absolute Error (MAE) methods, which are among the model evaluation criteria selected for the project, indicates that the prediction performance of the model is higher. On the other hand, in the R-Squared model evaluation metric, the results are between 1 and 0, and a higher value indicates higher model performance.
In this project, various machine learning models were trained and the models were tested with model evaluation criteria. As a result of the values obtained with the model evaluation criteria, various comparisons and inferences were made. The results obtained with the Mean Squared Error, Root Mean Squared Error and Mean Absolute Error model evaluation criteria show that the XGBoost model has the highest performance.

• XGBoost Mean Squared Error: 0.11515 

• XGBoost RMSE: 0.33934 

• XGBoost Mean Absolute Error: 0.24073

In common with the Mean Squared Error, Root Mean Squared Error and Mean Absolute Error model evaluation criteria, that measure how far off a regression model's predictions are from the actual values, the model with the lowest performance was observed to be Linear Regression.

• Linear Regression Mean Squared Error: 0.67919 

• Linear Regression RMSE: 0.82413 

• Linear Regression Mean Absolute Error: 0.60649

The R-square model evaluation method, which measures how well a regression model explains the variability in the data, takes values between 0 and 1, and the closer it is to 1, the better the model explains the variability in the data set.
The most significant results are presented below, showcasing the R-Squared metric's evaluation of the model's explanatory power concerning the variability in the data : 

• Linear Regression R^2 Score: 0.85202 

• Random Forest R^2 Score: 0.95515 

• XGBoost R^2 Score: 0.97491

It has been observed that one of the criteria with a high success rate in model evaluation criteria is the Random Forest model. Random Forest model evaluations for different models are listed below. 

• Random Forest Mean Squared Error: 0.19687 

• Random Forest RMSE: 0.45372 

• Random Forest R^2 Score: 0.95515 

• Random Forest Mean Absolute Error: 0.32069

In the model evaluation studies carried out after the established models are trained, metrics that can better adapt to the data and make more accurate predictions are observed. When evaluated, both XGBoost and Random Forest models consistently demonstrate good performance, and drawing attention with low error rates across various metrics such as R^2 score, RMSE, MSE, and MAE. On the other hand, the Linear Regression model shows higher error values, especially in metrics such as MSE, RMSE and MAE. This indicates that the accuracy of this model in explaining the data and making predictions in this project is lower than other models. This evaluation reveals the most appropriate model for this particular problem and this data set. It is evident that XGBoost and Random Forest models provide more robust results and have have strong performance.

### CONCLUSION
In the project of estimating the duration of hospital stay of patients, a literature review was conducted after the problem statement was made and relevant studies were examined. The methodologies and established models in the literature study and related studies were analyzed. A study process was carried out with analyzes based on the experiences gained by examining the relevant studies and the experiences gained from many studies in the literature review. After selecting the data set suitable for the project requirements, the correlations detected through the analyzes performed on the data set are among the important points that stand out in this project. While analyzing the data set in the project, various detailed aspects were examined and the inferences revealed showed the importance of studying this subject in machine learning in the literature.

The data set studied in this project has various strengths and weaknesses compared to other studies in the literature. The dataset studied in this project includes OECD countries for which all data for the years 1990-2018 are available at the same time. On the other hand, it was observed that other studies examined in the literature worked with data sets with narrower limits. The data set studied in the project is stronger than other studies in the literature in that it produces a very comprehensive study. On the other hand, considering the studies in the literature that work with a more limited data set, adopting a more focused approach may be a weakness of this project.
Various models established to predict the length of hospital stay of patients and the metrics by which these models are evaluated are quite powerful. In the literature review conducted before the model building study in the project, the weaknesses and strengths of existing projects and the results of various models were examined. The model evaluation results in the studies examined enabled this project to be worked on stronger foundations and put it ahead of other projects. The models used in the project and the model evaluation studies carried out as a result of training these models have shown the power of the project in terms of high accuracy rate.

During the literature review and review of related studies for the project, it was observed that there were not enough studies on machine learning technology in this field. A limited number of various studies have been examined, but compared to other studies in the field of machine learning, it is observed that studies in the field of predicting the length of hospital stay of patients and improving healthcare services are weak. The methodology used in this study, the correlations determined with the comprehensive data set studied, model building studies, model evaluation studies and the results obtained will be an important resource for future studies in this field. This project can be improved in the future with improvements in independent
variables and may provide prediction results closer to real values. With the improvements in the independent variables studied in the project, it is anticipated that more successful prediction results will be obtained in healthcare services that have many parameters, and this will provide significant benefits for patients with the increase in the quality of healthcare services.
