package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.model.LogisticRegression;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;

public class LogisticRegressionTest {

    private static final int nFeatures = 3;
    private static final double[] sigma = new double[] {1.0, 1.0, 1.0};
    private static final DoubleTensor trueWeights = DoubleTensor.create(new double[] {0.5, -3.0, 1.5}, new int[] {3, 1});
    private static final Double trueIntercept = 5.0;
    private static final int nSamplesTrain = 1000;
    private static final int nSamplesTest = 100;

    private DoubleTensor xTrain = generateX(nSamplesTrain);
    private DoubleTensor yTrain = generateY(xTrain);
    private DoubleTensor xTest = generateX(nSamplesTest);
    private DoubleTensor yTest = generateY(xTest);

    @Test
    public void testLogisticRegression() {
        LogisticRegression model = new LogisticRegression(xTrain, yTrain);
        model.fit();
        checkPredictions(model, 0.9, 0.85);
    }

    @Test
    public void testRegularizedLogisticRegression() {
        LogisticRegression regularizedModel = new LogisticRegression(xTrain, yTrain, 5.0);
        regularizedModel.fit();
        checkPredictions(regularizedModel, 0.9, 0.85);
    }

    private void checkPredictions(LogisticRegression model, double thresholdTrain, double thresholdTest) {
        DoubleTensor yHatTrain = model.predict(xTrain).round();
        DoubleTensor yHatTest = model.predict(xTest).round();
        assertAccurate(yHatTrain, yTrain, thresholdTrain);
        assertAccurate(yHatTest, yTest, thresholdTest);
    }

    private static DoubleTensor generateX(int nSamples) {
        DoubleVertex xVertex = new GaussianVertex(new int[] {nSamples, 1}, 0.0, sigma[0]);
        for (int i = 1; i < nFeatures; i++) {
            xVertex = new ConcatenationVertex(
                1, xVertex, new GaussianVertex(new int[] {nSamples, 1}, 0.0, sigma[i])
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

    private void assertAccurate(DoubleTensor yHat, DoubleTensor y, double threshold) {
        double accuracy = 1.0 - yHat.minus(y).abs().average();
        assertTrue(
            String.format("Observed accuracy %.2f is below the required threshold %.2f", accuracy, threshold),
            accuracy >= threshold
        );
        System.out.println(String.format("Observed accuracy was %.2f", accuracy));
    }
}
