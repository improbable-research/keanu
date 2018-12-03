package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.mcmc.proposal.GaussianProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
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
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.lessThan;

public class LinearRidgeRegressionTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    @Category(Slow.class)
    @Test
    public void findsParamsForOneWeight() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateSingleFeatureData();

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.RIDGE)
            .withPriorOnIntercept(0, 40)
            .build();

        linearRegressionModel.fit();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeightVertex(),
            linearRegressionModel.getInterceptVertex(),
            data
        );
    }

    @Category(Slow.class)
    @Test
    public void findsParamsForTwoWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateTwoFeatureData();
        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.RIDGE)
            .withPriorOnIntercept(0, 40)
            .build();

        linearRegressionModel.fit();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeightVertex(),
            linearRegressionModel.getInterceptVertex(),
            data
        );
    }

    @Test
    public void findsParamsForManyWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataGaussianWeights(20);

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.RIDGE)
            .withPriorOnIntercept(0, 40)
            .build();

        linearRegressionModel.fit();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeightVertex(),
            linearRegressionModel.getInterceptVertex(),
            data
        );
    }

    @Category(Slow.class)
    @Test
    public void decreasingSigmaDecreasesL2NormOfWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataGaussianWeights(20);

        RegressionModel linearRegressionModelWide = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.RIDGE)
            .withPriorOnWeightsAndIntercept(0, 100000)
            .build();
        RegressionModel linearRegressionModelNarrow = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.RIDGE)
            .withPriorOnWeightsAndIntercept(0, 0.00001)
            .build();

        linearRegressionModelWide.fit();

        linearRegressionModelNarrow.fit();

        assertThat(linearRegressionModelNarrow.getWeightVertex().getValue().pow(2).sum(), lessThan(linearRegressionModelWide.getWeightVertex().getValue().pow(2).sum()));

    }

    @Category(Slow.class)
    @Test
    public void youCanChooseSamplingInsteadOfGradientOptimization() {
        final int smallRawDataSize = 20;
        final int samplingCount = 30000;

        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateSingleFeatureData(smallRawDataSize);

        ProposalDistribution proposalDistribution = new GaussianProposalDistribution(DoubleTensor.scalar(0.25));

        SamplingModelFitting sampling = new SamplingModelFitting(MetropolisHastings.builder()
            .proposalDistribution(proposalDistribution)
            .variableSelector(MHStepVariableSelector.SINGLE_VARIABLE_SELECTOR)
            .build(),
            samplingCount);

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.RIDGE)
            .withPriorOnIntercept(0, data.intercept)
            .withPriorOnWeights(
                DoubleTensor.create(0., data.weights.getShape()),
                data.weights
            )
            .withSampling(sampling)
            .build();

        linearRegressionModel.fit();

        NetworkSamples networkSamples = sampling.getNetworkSamples().drop(samplingCount - 10000).downSample(100);

        assertSampledWeightsAndInterceptMatchTestData(
            networkSamples.getDoubleTensorSamples(linearRegressionModel.getWeightVertex().getId()),
            networkSamples.getDoubleTensorSamples(linearRegressionModel.getInterceptVertex().getId()),
            data);
    }
}
