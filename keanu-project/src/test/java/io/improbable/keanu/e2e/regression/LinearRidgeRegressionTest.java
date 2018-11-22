package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.model.regression.RegressionModel;
import io.improbable.keanu.model.regression.RegressionRegularization;
import io.improbable.keanu.testcategory.Slow;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;

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
            linearRegressionModel.getWeights(),
            linearRegressionModel.getIntercept(),
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
            linearRegressionModel.getWeights(),
            linearRegressionModel.getIntercept(),
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
            linearRegressionModel.getWeights(),
            linearRegressionModel.getIntercept(),
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

        assertThat(linearRegressionModelNarrow.getWeights().pow(2).sum(), lessThan(linearRegressionModelWide.getWeights().pow(2).sum()));

    }

}
