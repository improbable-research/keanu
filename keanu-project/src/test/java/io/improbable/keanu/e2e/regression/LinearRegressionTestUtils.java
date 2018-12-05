package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.tensor.BivariateDataStatisticsCalculator;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import lombok.Value;
import lombok.experimental.UtilityClass;
import lombok.extern.slf4j.Slf4j;

import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorMatchers.allCloseTo;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;

@UtilityClass
@Slf4j
class LinearRegressionTestUtils {
    private static int N = 100000;
    private static double EXPECTED_W1 = 3.0;
    private static double EXPECTED_W2 = 7.0;
    private static double EXPECTED_B = 20.0;

    static TestData generateSingleFeatureData() {
        return generateSingleFeatureData(N);
    }

    static TestData generateSingleFeatureData(int numSamples) {
        DoubleVertex xGenerator = new UniformVertex(new long[]{numSamples, 1}, 0, 10);
        DoubleVertex mu = xGenerator.multiply(EXPECTED_W1).plus(EXPECTED_B);
        DoubleVertex yGenerator = new GaussianVertex(mu, 1.0);
        DoubleTensor xData = xGenerator.sample();
        xGenerator.setAndCascade(xData);
        DoubleTensor yData = yGenerator.sample();

        return new TestData(DoubleTensor.scalar(EXPECTED_W1), EXPECTED_B, xData, yData);
    }

    static TestData generateTwoFeatureData() {
        return generateTwoFeatureData(N);
    }

    static TestData generateTwoFeatureData(int numSamples) {
        DoubleVertex x1Generator = new UniformVertex(new long[]{numSamples, 1}, 0, 10);
        DoubleVertex x2Generator = new UniformVertex(new long[]{numSamples, 1}, 50, 100);
        DoubleVertex yGenerator = new GaussianVertex(
            x1Generator.multiply(EXPECTED_W1).plus(x2Generator.multiply(EXPECTED_W2)).plus(EXPECTED_B),
            1.0
        );

        DoubleTensor x1Data = x1Generator.sample();
        x1Generator.setAndCascade(x1Data);
        DoubleTensor x2Data = x1Generator.sample();
        x2Generator.setAndCascade(x2Data);
        DoubleTensor yData = yGenerator.sample();

        return new TestData(DoubleTensor.create(EXPECTED_W1, EXPECTED_W2), EXPECTED_B, DoubleTensor.concat(1, x1Data, x2Data), yData);
    }

    static TestData generateThreeFeatureDataWithOneUncorrelatedFeature() {
        DoubleVertex x1Generator = new UniformVertex(new long[]{N, 1}, 0, 10);
        DoubleVertex x2Generator = new UniformVertex(new long[]{N, 1}, 50, 100);
        DoubleVertex x3Generator = new UniformVertex(new long[]{N, 1}, 50, 100);
        DoubleVertex yGenerator = new GaussianVertex(
            x1Generator.multiply(EXPECTED_W1).plus(x2Generator.multiply(EXPECTED_W2)).plus(EXPECTED_B),
            1.0
        );

        DoubleTensor x1Data = x1Generator.sample();
        x1Generator.setAndCascade(x1Data);
        DoubleTensor x2Data = x1Generator.sample();
        x2Generator.setAndCascade(x2Data);
        DoubleTensor yData = yGenerator.sample();

        return new TestData(DoubleTensor.create(EXPECTED_W1, EXPECTED_W2), EXPECTED_B, DoubleTensor.concat(1, x1Data, x2Data, x3Generator.sample()), yData);
    }

    static TestData generateMultiFeatureDataUniformWeights(int featureCount) {
        return generateMultiFeatureData(featureCount, shape -> new UniformVertex(shape, -10, 10));
    }

    static TestData generateMultiFeatureDataGaussianWeights(int featureCount) {
        return generateMultiFeatureData(featureCount, shape -> new GaussianVertex(shape, 0, 5));
    }

    static TestData generateMultiFeatureDataLaplaceWeights(int featureCount) {
        return generateMultiFeatureData(featureCount, shape -> new LaplaceVertex(shape, 0., 5.));
    }

    static TestData generateMultiFeatureData(int featureCount, Function<long[], DoubleVertex> weightVertexFromShape) {
        long N = 1000;
        double expectedB = 20;

        DoubleVertex xGenerator = new UniformVertex(new long[]{N, featureCount}, 0, 100);
        DoubleVertex weightsGenerator = weightVertexFromShape.apply(new long[]{featureCount,1});
        DoubleVertex yGenerator = new GaussianVertex(new long[]{N, 1}, xGenerator.matrixMultiply(weightsGenerator).plus(expectedB), 1.0);
        DoubleTensor xData = xGenerator.sample();
        DoubleTensor weights = weightsGenerator.sample();
        xGenerator.setValue(xData);
        weightsGenerator.setValue(weights);
        DoubleTensor yData = yGenerator.getValue();

        return new TestData(weights, expectedB, xData, yData);
    }

    static void assertWeightsAndInterceptMatchTestData(DoubleVertex weights, DoubleVertex intercept, TestData testData) {
        assertThat("Intercept", testData.intercept, closeTo(intercept.getValue().scalar(), 0.5));
        assertThat("Weights", weights.getValue(), allCloseTo(0.05, testData.weights));
    }

    static void assertSampledWeightsAndInterceptMatchTestData(DoubleVertexSamples gradient, DoubleVertexSamples intercept, TestData testData) {
        BivariateDataStatisticsCalculator statisticsCalculator = new BivariateDataStatisticsCalculator(testData.xTrain, testData.yTrain);
        double estimatedGradientInSampleData = statisticsCalculator.estimatedGradient();
        double estimatedInterceptInSampleData = statisticsCalculator.estimatedIntercept();
        double standardErrorForGradientInSampleData = statisticsCalculator.standardErrorForGradient();
        double standardErrorForInterceptInSampleData = statisticsCalculator.standardErrorForIntercept();

        log.info(String.format("Gradient from sampling:     %.4f ~ %.4f", gradient.getAverages().scalar(), gradient.getVariances().sqrt().scalar()));
        log.info(String.format("Gradient from initial data: %.4f ~ %.4f", estimatedGradientInSampleData, standardErrorForGradientInSampleData));
        log.info(String.format("Intercept from sampling:     %.4f ~ %.4f", intercept.getAverages().scalar(), intercept.getVariances().sqrt().scalar()));
        log.info(String.format("Intercept from initial data: %.4f ~ %.4f", estimatedInterceptInSampleData, standardErrorForInterceptInSampleData));

        assertThat(gradient.getAverages().scalar(), closeTo(estimatedGradientInSampleData, standardErrorForGradientInSampleData));
        assertThat(gradient.getVariances().sqrt().scalar(), closeTo(standardErrorForGradientInSampleData, 0.01));
        assertThat(intercept.getAverages().scalar(), closeTo(estimatedInterceptInSampleData, standardErrorForInterceptInSampleData));
        assertThat(intercept.getVariances().sqrt().scalar(), closeTo(standardErrorForInterceptInSampleData, 0.05));
    }

    @Value
    static class TestData {
        public DoubleTensor weights;
        public double intercept;
        public DoubleTensor xTrain;
        public DoubleTensor yTrain;
    }
}
