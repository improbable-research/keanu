package io.improbable.keanu.e2e.regression;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.lessThan;
import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.e2e.regression.LinearRegressionTestUtils.assertWeightsAndInterceptMatchTestData;
import static io.improbable.keanu.e2e.regression.LinearRegressionTestUtils.generateThreeFeatureDataWithOneUncorrelatedFeature;

import org.junit.Rule;
import org.junit.Test;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.model.regression.LinearLassoRegression;
import io.improbable.keanu.model.regression.LinearRegressionModel;

public class LinearLassoRegressionTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();


    @Test
    public void findsExpectedParamsForOneWeight() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateSingleFeatureData();

        LinearRegressionModel regression = LinearLassoRegression
            .withFeatureShape(data.xTrain.getShape())
            .setPriorOnIntercept(0, 40)
            .build();
        regression.fit(data.xTrain, data.yTrain);

        assertWeightsAndInterceptMatchTestData(
            regression.getWeights(),
            regression.getIntercept(),
            data
        );
    }

    @Test
    public void findsExpectedParamsForTwoWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateTwoFeatureData();
        LinearRegressionModel regression = LinearLassoRegression
            .withFeatureShape(data.xTrain.getShape())
            .setPriorOnIntercept(0, 40)
            .build();
        regression.fit(data.xTrain, data.yTrain);

        assertWeightsAndInterceptMatchTestData(
            regression.getWeights(),
            regression.getIntercept(),
            data
        );
    }

    @Test
    public void findsExpectedParamsForManyWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataLaplaceWeights(20);

        LinearRegressionModel regression = LinearLassoRegression
            .withFeatureShape(data.xTrain.getShape())
            .setPriorOnIntercept(0, 40)
            .build();
        regression.fit(data.xTrain, data.yTrain);

        assertWeightsAndInterceptMatchTestData(
            regression.getWeights(),
            regression.getIntercept(),
            data
        );
    }

    @Test
    public void decreasingSigmaDecreasesL1NormOfWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataLaplaceWeights(20);

        LinearRegressionModel regressionWide = LinearLassoRegression
            .withFeatureShape(data.xTrain.getShape())
            .setPriorOnIntercept(0, 100000)
            .build();
        LinearRegressionModel regressionNarrow = LinearLassoRegression
            .withFeatureShape(data.xTrain.getShape())
            .setPriorOnWeightsAndIntercept(0, 0.00001)
            .build();
        regressionWide.fit(data.xTrain, data.yTrain);
        regressionNarrow.fit(data.xTrain, data.yTrain);

        assertThat(regressionNarrow.getWeights().abs().sum(), lessThan(regressionWide.getWeights().abs().sum()));

    }

    @Test
    public void bringsUncorrelatedWeightsVeryCloseToZero() {
        LinearRegressionTestUtils.TestData data = generateThreeFeatureDataWithOneUncorrelatedFeature();

        LinearRegressionModel regression = LinearLassoRegression
            .withFeatureShape(data.xTrain.getShape())
            .build();

        regression.fit(data.xTrain, data.yTrain);

        assertEquals(0., regression.getWeight(2), 1e-3);
    }

}
