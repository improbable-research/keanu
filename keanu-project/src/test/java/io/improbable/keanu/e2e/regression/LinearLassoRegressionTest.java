package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.mcmc.proposal.GaussianProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.model.SamplingModelFitting;
import io.improbable.keanu.model.regression.RegressionModel;
import io.improbable.keanu.model.regression.RegressionRegularization;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import static io.improbable.keanu.e2e.regression.LinearRegressionTestUtils.assertSampledWeightsAndInterceptMatchTestData;
import static io.improbable.keanu.e2e.regression.LinearRegressionTestUtils.assertWeightsAndInterceptMatchTestData;
import static io.improbable.keanu.e2e.regression.LinearRegressionTestUtils.generateThreeFeatureDataWithOneUncorrelatedFeature;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.lessThan;

public class LinearLassoRegressionTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    @Category(Slow.class)
    @Test
    public void findsExpectedParamsForOneWeight() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateSingleFeatureData();

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.LASSO)
            .withPriorOnIntercept(0, 20)
            .build();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeights(),
            linearRegressionModel.getIntercept(),
            data
        );
    }

    @Category(Slow.class)
    @Test
    public void findsExpectedParamsForTwoWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateTwoFeatureData();
        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.LASSO)
            .withPriorOnIntercept(0, 20)
            .build();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeights(),
            linearRegressionModel.getIntercept(),
            data
        );
    }

    @Test
    public void findsExpectedParamsForManyWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataLaplaceWeights(20);

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.LASSO)
            .withPriorOnIntercept(0, 20)
            .build();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeights(),
            linearRegressionModel.getIntercept(),
            data
        );
    }

    @Category(Slow.class)
    @Test
    public void decreasingSigmaDecreasesL1NormOfWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataLaplaceWeights(20);

        RegressionModel linearRegressionModelWide = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.LASSO)
            .withPriorOnIntercept(0, 100000)
            .build();
        RegressionModel linearRegressionModelNarrow = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.LASSO)
            .withPriorOnWeightsAndIntercept(0, 0.00001)
            .build();

        assertThat(linearRegressionModelNarrow.getWeights().abs().sum(), lessThan(linearRegressionModelWide.getWeights().abs().sum()));

    }

    @Category(Slow.class)
    @Test
    public void bringsUncorrelatedWeightsVeryCloseToZero() {
        LinearRegressionTestUtils.TestData data = generateThreeFeatureDataWithOneUncorrelatedFeature();

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.LASSO)
            .build();

        assertThat(linearRegressionModel.getWeight(2), closeTo(0., 1e-3));
    }


    @Test
    public void youCanChooseSamplingInsteadOfGradientOptimization() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataLaplaceWeights(2);

        ProposalDistribution proposalDistribution = new GaussianProposalDistribution(DoubleTensor.scalar(.1));

        SamplingModelFitting sampling = new SamplingModelFitting(MetropolisHastings.builder()
            .proposalDistribution(proposalDistribution)
            .build(),
            3000);

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.LASSO)
            .withPriorOnIntercept(data.intercept, data.intercept * 0.01)
            .withPriorOnWeights(
                data.weights.asFlatDoubleArray(),
                data.weights.times(0.001).asFlatDoubleArray()
            )
            .withSampling(sampling)
            .build();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeights(),
            linearRegressionModel.getIntercept(),
            data
        );

        NetworkSamples networkSamples = sampling.getNetworkSamples().drop(2500).downSample(2);

        assertSampledWeightsAndInterceptMatchTestData(
            networkSamples.getDoubleTensorSamples(linearRegressionModel.getWeightsVertexId()),
            networkSamples.getDoubleTensorSamples(linearRegressionModel.getInterceptVertexId()),
            data);
    }

}
