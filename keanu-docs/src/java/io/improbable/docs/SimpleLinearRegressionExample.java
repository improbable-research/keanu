package io.improbable.docs;

import io.improbable.keanu.model.regression.RegressionModel;
import io.improbable.keanu.model.regression.RegressionRegularization;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class SimpleLinearRegressionExample {
    public static void main(String[] args) {
//%%SNIPPET_START%% SimpleLinearRegressionExample
//Define Model Parameters
double weight = 2.0;
double offset = 20.0;
int numberOfSamples = 100;

//Define random input data vertex by sampling from uniform probability distribution between 0 and 10
DoubleVertex xGenerator = new UniformVertex(new long[]{1, numberOfSamples}, 0, 10);

//Define the desired output vertices
DoubleVertex yMu = xGenerator.multiply(weight).plus(offset);

//Define a vertex for taking noisy readings of output data
DoubleVertex yGenerator = new GaussianVertex(yMu, 1.0);

//Sample input data and then sample the corresponding noisy output data
DoubleTensor xData = xGenerator.sample();
xGenerator.setAndCascade(xData);
DoubleTensor yData = yGenerator.sample();

//Create a simple linear regression model and fit it with our input and output data
RegressionModel regressionModel = RegressionModel.withTrainingData(xData, yData)
    .withRegularization(RegressionRegularization.NONE)
    .build();

//It is now possible to use regressionModel.predict(value) to get a prediction of the output given an input value.
//%%SNIPPET_END%% SimpleLinearRegressionExample
    }
}