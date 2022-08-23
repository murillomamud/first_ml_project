### First Machine Learning Study

In this repository exists a simple first machine learning model construction using the file churn_modelling to predict if a customer will leave or continue in the bank.

I created a Dockerfile file to make easy run it in your computer. Installing tensorflow can be a little hard, so, to execute this file do the following.

Access the folder with the project and:

```
docker build -t imagename .
docker run -it --rm -v $PWD:/tmp -w /tmp imagename python ./main.py
```