package io.improbable.keanu.e2e.regression;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.lessThan;

import static io.improbable.keanu.e2e.regression.LinearRegressionTestUtils.assertWeightsAndInterceptMatchTestData;

import org.junit.Rule;
import org.junit.Test;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.model.regression.LinearRegressionModel;
import io.improbable.keanu.model.regression.LinearRidgeRegression;

public class LinearRidgeRegressionTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();


    @Test
    public void findsParamsForOneWeight() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateSingleFeatureData();

        LinearRegressionModel regression = LinearRidgeRegression
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
    public void findsParamsForTwoWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateTwoFeatureData();
        LinearRegressionModel regression = LinearRidgeRegression
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
    public void findsParamsForManyWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataGaussianWeights(20);

        LinearRegressionModel regression = LinearRidgeRegression
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
    public void decreasingSigmaDecreasesL2NormOfWeights() {
        LinearRegressionTestUtils.TestData data = LinearRegressionTestUtils.generateMultiFeatureDataGaussianWeights(20);

        LinearRegressionModel regressionWide = LinearRidgeRegression
            .withFeatureShape(data.xTrain.getShape())
            .setPriorOnWeightsAndIntercept(0, 100000)
            .build();
        LinearRegressionModel regressionNarrow = LinearRidgeRegression
            .withFeatureShape(data.xTrain.getShape())
            .setPriorOnWeightsAndIntercept(0, 0.00001)
            .build();
        regressionWide.fit(data.xTrain, data.yTrain);
        regressionNarrow.fit(data.xTrain, data.yTrain);

        assertThat(regressionNarrow.getWeights().pow(2).sum(), lessThan(regressionWide.getWeights().pow(2).sum()));

    }

}
