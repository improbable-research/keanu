package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.mcmc.RollbackAndCascadeOnRejection;
import io.improbable.keanu.algorithms.mcmc.proposal.GaussianProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.model.SamplingModelFitting;
import io.improbable.keanu.model.regression.RegressionModel;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import static io.improbable.keanu.e2e.regression.LinearRegressionTestUtils.assertSampledWeightsAndInterceptMatchTestData;
import static io.improbable.keanu.e2e.regression.LinearRegressionTestUtils.assertWeightsAndInterceptMatchTestData;

public class LinearRegressionTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    @Category(Slow.class)
    @Test
    public void manuallyBuiltGraphFindsParamsForOneWeight() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateSingleFeatureData();

        DoubleVertex weight = new GaussianVertex(0, 10.0);
        DoubleVertex intercept = new GaussianVertex(0, 10.0);
        DoubleVertex x = ConstantVertex.of(data.xTrain);
        DoubleVertex y = new GaussianVertex(x.multiply(weight).plus(intercept), 5.0);
        y.observe(data.yTrain);

        io.improbable.keanu.algorithms.variational.optimizer.Optimizer optimizer = Keanu.Optimizer.of(weight.getConnectedGraph());
        optimizer.maxLikelihood();

        assertWeightsAndInterceptMatchTestData(
            weight,
            intercept,
            data
        );
    }

    @Category(Slow.class)
    @Test
    public void modelFindsParamsForOneWeight() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateSingleFeatureData();

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .build();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeightVertex(),
            linearRegressionModel.getInterceptVertex(),
            data
        );
    }

    @Category(Slow.class)
    @Test
    public void manuallyBuiltGraphFindsParamsForTwoWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateTwoFeatureData();

        DoubleVertex w1 = new GaussianVertex(0.0, 10.0);
        DoubleVertex w2 = new GaussianVertex(0.0, 10.0);
        DoubleVertex b = new GaussianVertex(0.0, 10.0);
        DoubleVertex x1 = ConstantVertex.of(data.xTrain.slice(1, 0).reshape(100000, 1));
        DoubleVertex x2 = ConstantVertex.of(data.xTrain.slice(1, 1).reshape(100000, 1));
        DoubleVertex y = new GaussianVertex(x1.multiply(w1).plus(x2.multiply(w2)).plus(b), 5.0);
        y.observe(data.yTrain);

        BayesianNetwork bayesNet = new BayesianNetwork(y.getConnectedGraph());
        GradientOptimizer optimizer = Keanu.Optimizer.Gradient.of(bayesNet);

        optimizer.maxLikelihood();

        assertWeightsAndInterceptMatchTestData(
            ConstantVertex.of(DoubleTensor.concat(0, w1.getValue(), w2.getValue())),
            b,
            data
        );
    }

    @Category(Slow.class)
    @Test
    public void modelFindsParamsForTwoWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateTwoFeatureData();
        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .build();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeightVertex(),
            linearRegressionModel.getInterceptVertex(),
            data
        );
    }

    @Category(Slow.class)
    @Test
    public void manuallyBuiltGraphFindsParamsForManyWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataUniformWeights(40);

        DoubleVertex weights = new GaussianVertex(new long[]{40, 1}, 0, 1);
        DoubleVertex intercept = new GaussianVertex(0, 1);
        DoubleVertex x = ConstantVertex.of(data.xTrain);
        DoubleVertex y = new GaussianVertex(x.matrixMultiply(weights).plus(intercept), 1);
        y.observe(data.yTrain);

        BayesianNetwork bayesNet = new BayesianNetwork(y.getConnectedGraph());
        GradientOptimizer optimizer = Keanu.Optimizer.Gradient.of(bayesNet);
        optimizer.maxLikelihood();

        assertWeightsAndInterceptMatchTestData(
            weights,
            intercept,
            data
        );
    }

    @Test
    public void modelFindsParamsForManyWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataUniformWeights(20);

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .build();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeightVertex(),
            linearRegressionModel.getInterceptVertex(),
            data
        );
    }

    @Category(Slow.class)
    @Test
    public void youCanChooseSamplingInsteadOfGradientOptimization() {
        final int smallRawDataSize = 20;
        final int samplingCount = 6000;

        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateSingleFeatureData(smallRawDataSize);

        ProposalDistribution proposalDistribution = new GaussianProposalDistribution(DoubleTensor.scalar(0.25));

        SamplingModelFitting sampling = new SamplingModelFitting(model -> MetropolisHastings.builder()
            .proposalDistribution(proposalDistribution)
            .variableSelector(MHStepVariableSelector.SINGLE_VARIABLE_SELECTOR)
            .rejectionStrategy(new RollbackAndCascadeOnRejection())
            .build(),
            samplingCount);

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withPriorOnIntercept(0, data.intercept)
            .withPriorOnWeights(
                DoubleTensor.create(0., data.weights.getShape()),
                data.weights
            )
            .withSampling(sampling)
            .build();

        NetworkSamples networkSamples = sampling.getNetworkSamples().drop(samplingCount / 10).downSample(2);

        assertSampledWeightsAndInterceptMatchTestData(
            networkSamples.getDoubleTensorSamples(linearRegressionModel.getWeightVertex().getId()),
            networkSamples.getDoubleTensorSamples(linearRegressionModel.getInterceptVertex().getId()),
            data);
    }
}