package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.model.LogisticRegression;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;

public class LogisticRegressionTest {

    private static final int numFeatures = 3;
    private static final double[] sigmas = new double[] {1.0, 1.0, 1.0};

    private static final DoubleTensor trueWeights = DoubleTensor.create(new double[] {0.5, -3.0, 1.5}, 3, 1);
    private static final double trueIntercept = 5.0;

    private static final int numSamplesForTraining = 1000;
    private static final int numSamplesForTesting = 100;

    private DoubleTensor xTrain = generateX(numSamplesForTraining);
    private DoubleTensor yTrain = generateY(xTrain);
    private DoubleTensor xTest = generateX(numSamplesForTesting);
    private DoubleTensor yTest = generateY(xTest);

    @Test
    public void testLogisticRegression() {
        LogisticRegression model = new LogisticRegression(xTrain, yTrain);
        model = model.fit();
        checkPredictedValuesAreAccurate(model, 0.9, 0.85);
    }

    @Test
    public void testRegularizedLogisticRegression() {
        LogisticRegression regularizedModel = new LogisticRegression(xTrain, yTrain, 5.0);
        regularizedModel = regularizedModel.fit();
        checkPredictedValuesAreAccurate(regularizedModel, 0.9, 0.85);
    }

    private void checkPredictedValuesAreAccurate(LogisticRegression model, double thresholdTrain, double thresholdTest) {
        DoubleTensor predictedYTrain = model.predict(xTrain).round();
        DoubleTensor predictedYTest = model.predict(xTest).round();
        assertYValuesArePredicted(predictedYTrain, yTrain, thresholdTrain);
        assertYValuesArePredicted(predictedYTest, yTest, thresholdTest);
    }

    private static DoubleTensor generateX(int nSamples) {
        DoubleVertex xVertex = new GaussianVertex(new int[] {nSamples, 1}, 0.0, sigmas[0]);
        for (int i = 1; i < numFeatures; i++) {
            xVertex = new ConcatenationVertex(
                1, xVertex, new GaussianVertex(new int[] {nSamples, 1}, 0.0, sigmas[i])
            );
        }
        return xVertex.sample();
    }

    private static DoubleTensor generateY(DoubleTensor x) {
        DoubleTensor probabilities = x.matrixMultiply(trueWeights)
            .plus(trueIntercept)
            .sigmoid();
        BoolVertex yVertex = new BernoulliVertex(new ConstantDoubleVertex(probabilities));
        double[] outcome = yVertex.sample().asFlatList().stream().mapToDouble(d -> d ? 1.0 : 0.0).toArray();
        return DoubleTensor.create(outcome, probabilities.getShape());
    }

    private void assertYValuesArePredicted(DoubleTensor predictedYValues, DoubleTensor y, double threshold) {
        double accuracy = 1.0 - predictedYValues.minus(y).abs().average();
        assertTrue(
            String.format("Observed accuracy %.2f is below the required threshold %.2f", accuracy, threshold),
            accuracy >= threshold
        );
    }
}
