package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.variational.tensor.TensorGradientOptimizer;
import io.improbable.keanu.network.BayesNetTensorAsContinuous;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorUniformVertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

public class LinearRegression {
    private final Logger log = LoggerFactory.getLogger(LinearRegression.class);

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void linearRegression1FactorTensorVariationalMAP() {

        // Generate data
        int N = 100000;
        double expectedM = 3.0;
        double expectedB = 20.0;

        DoubleTensorVertex xGenerator = new TensorUniformVertex(new int[]{1, N}, 0, 10);
        DoubleTensorVertex yGenerator = new TensorGaussianVertex(xGenerator.multiply(expectedM).plus(expectedB), 1.0);
        DoubleTensor xData = xGenerator.sample(random);
        xGenerator.setAndCascade(xData);
        DoubleTensor yData = yGenerator.sample(random);

        // Linear Regression
        DoubleTensorVertex m = new TensorGaussianVertex(0.0, 10.0);
        DoubleTensorVertex b = new TensorGaussianVertex(0.0, 10.0);
        DoubleTensorVertex x = new ConstantDoubleTensorVertex(xData);
        DoubleTensorVertex y = new TensorGaussianVertex(x.multiply(m).plus(b), 5.0);
        y.observe(yData);

        BayesNetTensorAsContinuous bayesNet = new BayesNetTensorAsContinuous(m.getConnectedGraph());
        TensorGradientOptimizer optimizer = new TensorGradientOptimizer(bayesNet);

        optimizer.maxLikelihood(10000);

        log.info("M = " + m.getValue().scalar() + ", B = " + b.getValue().scalar());
        assertEquals(expectedM, m.getValue().scalar(), 0.05);
        assertEquals(expectedB, b.getValue().scalar(), 0.05);
    }

    @Test
    public void linearRegressionTwoFactorTensorVariationalMAP() {

        // Generate data
        int N = 100000;
        double expectedW1 = 3.0;
        double expectedW2 = 7.0;
        double expectedB = 20.0;

        DoubleTensorVertex x1Generator = new TensorUniformVertex(new int[]{1, N}, 0, 10);
        DoubleTensorVertex x2Generator = new TensorUniformVertex(new int[]{1, N}, 50, 100);
        DoubleTensorVertex yGenerator = new TensorGaussianVertex(
            x1Generator.multiply(expectedW1).plus(x2Generator.multiply(expectedW2)).plus(expectedB),
            1.0
        );
        DoubleTensor x1Data = x1Generator.sample(random);
        x1Generator.setAndCascade(x1Data);
        DoubleTensor x2Data = x1Generator.sample(random);
        x2Generator.setAndCascade(x2Data);
        DoubleTensor yData = yGenerator.sample(random);

        // Linear Regression
        DoubleTensorVertex w1 = new TensorGaussianVertex(0.0, 10.0);
        DoubleTensorVertex w2 = new TensorGaussianVertex(0.0, 10.0);
        DoubleTensorVertex b = new TensorGaussianVertex(0.0, 10.0);
        DoubleTensorVertex x1 = new ConstantDoubleTensorVertex(x1Data);
        DoubleTensorVertex x2 = new ConstantDoubleTensorVertex(x2Data);
        DoubleTensorVertex y = new TensorGaussianVertex(x1.multiply(w1).plus(x2.multiply(w2)).plus(b), 5.0);
        y.observe(yData);

        BayesNetTensorAsContinuous bayesNet = new BayesNetTensorAsContinuous(y.getConnectedGraph());
        TensorGradientOptimizer optimizer = new TensorGradientOptimizer(bayesNet);

        optimizer.maxLikelihood(10000);

        log.info("W1 = " + w1.getValue().scalar() + " W2 = " + w2.getValue().scalar() + ", B = " + b.getValue().scalar());
        assertEquals(expectedW1, w1.getValue().scalar(), 0.05);
        assertEquals(expectedW2, w2.getValue().scalar(), 0.05);
        assertEquals(expectedB, b.getValue().scalar(), 0.05);
    }

}