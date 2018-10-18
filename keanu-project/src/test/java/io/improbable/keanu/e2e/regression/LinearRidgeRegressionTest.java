package io.improbable.keanu.e2e.regression;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.lessThan;

import static io.improbable.keanu.e2e.regression.LinearRegressionTestUtils.assertWeightsAndInterceptMatchTestData;

import io.improbable.keanu.model.regression.RegressionRegularization;
import org.junit.Rule;
import org.junit.Test;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.model.regression.RegressionModel;

public class LinearRidgeRegressionTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();


    @Test
    public void findsParamsForOneWeight() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateSingleFeatureData();

        RegressionModel linearRegressionModel = RegressionModel.builder()
            .setRegularization(RegressionRegularization.RIDGE)
            .setInputTrainingData(data.xTrain)
            .setOutputTrainingData(data.yTrain)
            .setPriorOnIntercept(0, 40)
            .buildLinearRegressionModel();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeights(),
            linearRegressionModel.getIntercept(),
            data
        );
    }

    @Test
    public void findsParamsForTwoWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateTwoFeatureData();
        RegressionModel linearRegressionModel = RegressionModel.builder()
            .setRegularization(RegressionRegularization.RIDGE)
            .setInputTrainingData(data.xTrain)
            .setOutputTrainingData(data.yTrain)
            .setPriorOnIntercept(0, 40)
            .buildLinearRegressionModel();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeights(),
            linearRegressionModel.getIntercept(),
            data
        );
    }

    @Test
    public void findsParamsForManyWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataGaussianWeights(20);

        RegressionModel linearRegressionModel = RegressionModel.builder()
            .setRegularization(RegressionRegularization.RIDGE)
            .setInputTrainingData(data.xTrain)
            .setOutputTrainingData(data.yTrain)
            .setPriorOnIntercept(0, 40)
            .buildLinearRegressionModel();

        assertWeightsAndInterceptMatchTestData(
            linearRegressionModel.getWeights(),
            linearRegressionModel.getIntercept(),
            data
        );
    }

    @Test
    public void decreasingSigmaDecreasesL2NormOfWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataGaussianWeights(20);

        RegressionModel linearRegressionModelWide = RegressionModel.builder()
            .setRegularization(RegressionRegularization.RIDGE)
            .setInputTrainingData(data.xTrain)
            .setOutputTrainingData(data.yTrain)
            .setPriorOnWeightsAndIntercept(0, 100000)
            .buildLinearRegressionModel();
        RegressionModel linearRegressionModelNarrow = RegressionModel.builder()
            .setRegularization(RegressionRegularization.RIDGE)
            .setInputTrainingData(data.xTrain)
            .setOutputTrainingData(data.yTrain)
            .setPriorOnWeightsAndIntercept(0, 0.00001)
            .buildLinearRegressionModel();

        assertThat(linearRegressionModelNarrow.getWeights().pow(2).sum(), lessThan(linearRegressionModelWide.getWeights().pow(2).sum()));

    }

}
