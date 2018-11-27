package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.model.regression.RegressionModel;
import io.improbable.keanu.model.regression.RegressionRegularization;
import io.improbable.keanu.testcategory.Slow;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;

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

}
