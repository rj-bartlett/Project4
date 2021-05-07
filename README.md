## Project 4

### Introduction:

	For this project, I want to see if I can use machine learning to predict the quality of different types of red wines based on their chemical components. This research topic is important because alcohol is a large part of the global economy and wine specifically has an important role in many lifestyles, especially through aspects such as the Mediterranean diet. Being able to find a way to qualify wine will help wineries make better wine if they know what characteristics to look for in terms of psychochemical components. In terms of machine learning, this problem has been an interest for data mining purposes as investigated in a 2015 study which performed four data mining/fuzzy models. Can I create a machine learning program that is similarly accurate to the accuracy of the model named in this 2015 study?
  
### Data:

	I obtained the data from the Kaggle database, although the data originated from a 2009 study. This dataset contains 1499 different observations of the Portuguese "Vinho Verde" red wines and their associated qualities (on a scale of 1-10). There are 11 other physicochemical descriptors besides quality which include fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, and alcohol content. Every variable except for quality are continuous variables. Quality is a discrete variable because it is limited in its range: 1-10. 

### Methodology:
	
  The code I used to set up the model and compile it can be found below. I decided to use a neural network model because in the paper it stated how neural networks have been able to predict wine quality with as much success as their models so I wanted to see how accurate I could get a basic neural network model using methods learned throughout this semester. 
	
  Since the result was not as successful as I had hoped, I am considering going back and using either a form of regression model or random forest model and compare my two models to the four models from the paper. It would be interesting to see compare models that we have learned in class to models created from “fuzzy techniques.” I also am not completely set on the setup of the neural networks because I think that they can be improved, with a little more research. 
	
  Making the model:

``` python 
model = tf.keras.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1024, activation=tf.nn.relu), 
  tf.keras.layers.Dense(10, activation = tf.nn.softmax)])
```

	Compiling the model:

``` python
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
```

### Conclusion:
	
  After training 1199 of the 1499 observations on the neural network model I created, the highest accuracy I could obtain was around 60%. The first epochs were around 40% and gradually increased from there. But once the accuracy got around 60%, the accuracy then started to fluctuate and remain around 60% which means that the model stopped learning. Before the final project deadline, I will try to understand more about the Dense layers and possible activation functions that could have given me a higher accuracy. On evaluating the model, the model had an accuracy for predicting of .63249. When I specifically compared the first 10 classifications to the actual quality of the wine, 8 of 10 had a correct result. 
	
  All of the models in the paper took significantly longer to run, from 10 minutes to almost 24 hours, but each of them were significantly better than the first model I have made so far, with accuracies almost in the 90%. Overall, the Fuzzy Inductive Reasoning (FIR) model was the most accurate at predicting wine quality than any of the other tests mentioned thus far. This model was also better than any of the models included in the original study, which included neural networks. 

Link to 2015 study: https://www.scitepress.org/Papers/2015/55519/55519.pdf

Link to data: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

