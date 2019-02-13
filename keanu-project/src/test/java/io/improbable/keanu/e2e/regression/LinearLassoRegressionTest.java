package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.mcmc.RollbackAndCascadeOnRejection;
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
            linearRegressionModel.getWeightVertex(),
            linearRegressionModel.getInterceptVertex(),
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

        linearRegressionModel.fit();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeightVertex(),
            linearRegressionModel.getInterceptVertex(),
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

        linearRegressionModel.fit();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeightVertex(),
            linearRegressionModel.getInterceptVertex(),
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

        linearRegressionModelWide.fit();

        linearRegressionModelNarrow.fit();

        assertThat(linearRegressionModelNarrow.getWeightVertex().getValue().abs().sum(), lessThan(linearRegressionModelWide.getWeightVertex().getValue().abs().sum()));

    }

    @Category(Slow.class)
    @Test
    public void bringsUncorrelatedWeightsVeryCloseToZero() {
        LinearRegressionTestUtils.TestData data = generateThreeFeatureDataWithOneUncorrelatedFeature();

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.LASSO)
            .build();

        linearRegressionModel.fit();

        assertThat(linearRegressionModel.getWeightVertex().getValue(2), closeTo(0., 1e-3));
    }

    @Category(Slow.class)
    @Test
    public void youCanChooseSamplingInsteadOfGradientOptimization() {
        final int smallRawDataSize = 20;
        final int samplingCount = 5000;

        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateSingleFeatureData(smallRawDataSize);

        ProposalDistribution proposalDistribution = new GaussianProposalDistribution(DoubleTensor.scalar(0.25));

        SamplingModelFitting sampling = new SamplingModelFitting(model -> MetropolisHastings.builder()
            .proposalDistribution(proposalDistribution)
            .variableSelector(MHStepVariableSelector.SINGLE_VARIABLE_SELECTOR)
            .rejectionStrategy(new RollbackAndCascadeOnRejection())
            .build(),
            samplingCount);

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .withRegularization(RegressionRegularization.LASSO)
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
