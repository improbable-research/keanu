package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.model.ModelScoring;
import io.improbable.keanu.model.regression.RegressionModel;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.allCloseTo;
import static org.hamcrest.MatcherAssert.assertThat;

public class LogisticRegressionTest {

    private static final int NUM_FEATURES = 3;
    private static final double[] SIGMAS = new double[]{1.0, 1.0, 1.0};
    private static final DoubleTensor TRUE_WEIGHTS = DoubleTensor.create(new double[]{0.5, -3.0, 1.5}, 3, 1);
    private static final double TRUE_INTERCEPT = 0.0;
    private static final int NUM_SAMPLES_TRAINING = 1250;
    private static final int NUM_SAMPLES_TESTING = 200;
    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();
    private DoubleTensor xTrain;
    private BooleanTensor yTrain;
    private DoubleTensor xTest;
    private BooleanTensor yTest;
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
        RegressionModel<BooleanTensor> model = RegressionModel.withTrainingData(xTrain, yTrain)
            .build();

        model.fit();
        double accuracy = ModelScoring.accuracy(model.predict(xTest), yTest);
        Assert.assertTrue(accuracy > 0.75);
        assertWeightsAreCalculated(model.getWeights());
    }

    private DoubleTensor generateX(int nSamples) {
        DoubleVertex[] xVertices = new DoubleVertex[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; i++) {
            xVertices[i] = new GaussianVertex(new long[]{nSamples, 1}, 0.0, SIGMAS[i]);
        }
        return DoubleVertex.concat(1, xVertices).sample(random);
    }

    private BooleanTensor generateY(DoubleTensor x) {
        DoubleTensor probabilities = x.matrixMultiply(TRUE_WEIGHTS).plus(TRUE_INTERCEPT).sigmoid();
        BoolVertex yVertex = new BernoulliVertex(ConstantVertex.of(probabilities));
        return yVertex.getValue();
    }

    private void assertWeightsAreCalculated(DoubleTensor weights) {
        assertThat(weights, allCloseTo(0.15, TRUE_WEIGHTS));
    }

}
