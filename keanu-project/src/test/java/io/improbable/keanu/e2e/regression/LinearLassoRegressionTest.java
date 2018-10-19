package io.improbable.keanu.e2e.regression;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.lessThan;
import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.e2e.regression.LinearRegressionTestUtils.assertWeightsAndInterceptMatchTestData;
import static io.improbable.keanu.e2e.regression.LinearRegressionTestUtils.generateThreeFeatureDataWithOneUncorrelatedFeature;

import io.improbable.keanu.model.regression.RegressionRegularization;
import org.junit.Rule;
import org.junit.Test;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.model.regression.RegressionModel;

public class LinearLassoRegressionTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();


    @Test
    public void findsExpectedParamsForOneWeight() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateSingleFeatureData();

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .setRegularization(RegressionRegularization.LASSO)
            .setPriorOnIntercept(0, 40)
            .build();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeights(),
            linearRegressionModel.getIntercept(),
            data
        );
    }

    @Test
    public void findsExpectedParamsForTwoWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateTwoFeatureData();
        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .setRegularization(RegressionRegularization.LASSO)
            .setPriorOnIntercept(0, 40)
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
            .setRegularization(RegressionRegularization.LASSO)
            .setPriorOnIntercept(0, 40)
            .build();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeights(),
            linearRegressionModel.getIntercept(),
            data
        );
    }

    @Test
    public void decreasingSigmaDecreasesL1NormOfWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataLaplaceWeights(20);

        RegressionModel linearRegressionModelWide = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .setRegularization(RegressionRegularization.LASSO)
            .setPriorOnIntercept(0, 100000)
            .build();
        RegressionModel linearRegressionModelNarrow = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .setRegularization(RegressionRegularization.LASSO)
            .setPriorOnWeightsAndIntercept(0, 0.00001)
            .build();

        assertThat(linearRegressionModelNarrow.getWeights().abs().sum(), lessThan(linearRegressionModelWide.getWeights().abs().sum()));

    }

    @Test
    public void bringsUncorrelatedWeightsVeryCloseToZero() {
        LinearRegressionTestUtils.TestData data = generateThreeFeatureDataWithOneUncorrelatedFeature();

        RegressionModel linearRegressionModel = RegressionModel.withTrainingData(data.xTrain, data.yTrain)
            .setRegularization(RegressionRegularization.LASSO)
            .build();

        assertEquals(0., linearRegressionModel.getWeight(2), 1e-3);
    }

}
