package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.distributions.gradient.Logistic;
import io.improbable.keanu.model.LogisticRegression;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;
import static org.hamcrest.MatcherAssert.assertThat;

public class LogisticRegressionTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    private static final int NUM_FEATURES = 3;
    private static final double[] SIGMAS = new double[] {1.0, 1.0, 1.0};

    private static final DoubleTensor TRUE_WEIGHTS = DoubleTensor.create(new double[] {0.5, -3.0, 1.5}, 1, 3);
    private static final double TRUE_INTERCEPT = 5.0;

    private static final int NUM_SAMPLES_TRAINING = 1250;
    private static final int NUM_SAMPLES_TESTING = 200;

    private DoubleTensor xTrain;
    private DoubleTensor yTrain;
    private DoubleTensor xTest;
    private DoubleTensor yTest;
    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        xTrain = generateX(NUM_SAMPLES_TRAINING);
        yTrain = generateY(xTrain);
        xTest = generateX(NUM_SAMPLES_TESTING);
        yTest = generateY(xTest);
    }

    @Test
    public void testLogisticRegression() {
        LogisticRegression model = new LogisticRegression(xTrain, yTrain);
        model = model.fit();
        double score = model.score(xTest, yTest);
        assertTrue(score > 0.3);
        assertWeightsAreCalculated(model.getWeights());
    }

    @Test
    public void testRegularizedLogisticRegression() {
        LogisticRegression unregularizedModel = new LogisticRegression(xTrain, yTrain);
        unregularizedModel = unregularizedModel.fit();

        LogisticRegression regularizedModel = new LogisticRegression(xTrain, yTrain, 5.);
        regularizedModel = regularizedModel.fit();

        assertRegularizedWeightsAreSmaller(unregularizedModel.getWeights(), regularizedModel.getWeights());
        double score = regularizedModel.score(xTest, yTest);
        assertTrue(score > 0.3);
        assertWeightsAreCalculated(regularizedModel.getWeights());
    }

    @Test(expected = RuntimeException.class)
    public void predictionFailsIfModelIsNotFit() {
        LogisticRegression model = new LogisticRegression(xTrain, yTrain);
        model.predict(xTest);
    }

    private DoubleTensor generateX(int nSamples) {
        DoubleVertex[] xVertices = new DoubleVertex[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; i++) {
            xVertices[i] = new GaussianVertex(new int[] {1, nSamples}, 0.0, SIGMAS[i]);
        }
        return DoubleVertex.concat(0, xVertices).sample(random);
    }

    private DoubleTensor generateY(DoubleTensor x) {
        DoubleTensor probabilities = TRUE_WEIGHTS.matrixMultiply(x).plus(TRUE_INTERCEPT).sigmoid();
        BoolVertex yVertex = new BernoulliVertex(ConstantVertex.of(probabilities));
        double[] outcome = yVertex.sample(random).asFlatList().stream().mapToDouble(d -> d ? 1.0 : 0.0).toArray();
        return DoubleTensor.create(outcome, probabilities.getShape());
    }

    private void assertWeightsAreCalculated(DoubleVertex weights) {
        assertTrue(weights.getValue().equalsWithinEpsilon(TRUE_WEIGHTS, 0.5));
    }

    private void assertRegularizedWeightsAreSmaller(DoubleVertex unregularizedWeights, DoubleVertex regularizedWeights) {
        assertTrue(regularizedWeights.getValue().abs().lessThanOrEqual(unregularizedWeights.getValue().abs()).allTrue());
    }
}
