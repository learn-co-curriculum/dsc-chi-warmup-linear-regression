
# Linear Regression

This morning's warmup will more a codealong.

Since we're nearing the end of mod project, let's make sure we have the code down for fitting a linear regression model!

Below we import life expectancy data for 193 countries from 2000-2015.

For this warmup, let's:
- Limit the dataset to the year 2015
- Remove all columns containing strings
- Lower the column names and replace spaces with underscores

Great, let's make sure we don't have any null values!

# $Yikes$

Ok, let's first drop the ```alcohol``` and ```total_expenditure``` columns.

Once we've done that we can drop any other rows that have null values. 

We will also drop our ```year``` column for now.

Next we need to isolate out X and y data.

For this dataset, the column ```life_expectancy``` is our target column *(our y column)*

Ok, now in the cell below import ```statsmodels.api as sm```.

And then we fit out model!

**How well is our model explaining the variance of our data?**

**Before we go, let's fit an sklearn model as well.**

First we create a model object.

Then we fit the model object on our X, and y data.

Next we use the model to predict values for our X data.

Then we compare our predictions to the actual y values

Now let's plot our coefficients!

There are some major differences between the coefficients. It might be beneficial to scale our data using Sklearn's Standard Scaler.

To speed things up, let's compile out modeling code into functions

No change! Let's take a look at the relationship between income_composition_of_resources and our target.

Let's see what happens to our coefficients if we drop this column.
