package io.improbable.keanu.model.regression;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex;

public enum RegressionRegularization {
    NONE {
        public DoubleVertex getWeightsVertex(long featureCount,  double[] priorOnWeightsMeans, double[] priorOnInterceptScaleParameter) {
            return new GaussianVertex(new long[]{1, featureCount}, ConstantVertex.of(DEFAULT_MU), ConstantVertex.of(DEFAULT_SCALE_PARAMETER));
        }
        public DoubleVertex getInterceptVertex(Double priorOnInterceptMean, Double priorOnInterceptScaleParameter) {
            return new GaussianVertex(DEFAULT_MU, DEFAULT_SCALE_PARAMETER);
        }
    },
    LASSO {
        public DoubleVertex getWeightsVertex(long featureCount, double[] priorOnWeightsMeans, double[] priorOnInterceptScaleParameter) {
            return new LaplaceVertex(new long[]{1, featureCount}, ConstantVertex.of(priorOnWeightsMeans), ConstantVertex.of(priorOnInterceptScaleParameter));
        }
        public DoubleVertex getInterceptVertex(Double priorOnInterceptMean, Double priorOnInterceptScaleParameter) {
            return new LaplaceVertex(priorOnInterceptMean, priorOnInterceptScaleParameter);
        }
    },
    RIDGE {
        public DoubleVertex getWeightsVertex(long featureCount, double[] priorOnWeightsMeans, double[] priorOnInterceptScaleParameter) {
            return new GaussianVertex(new long[]{1, featureCount}, ConstantVertex.of(priorOnWeightsMeans), ConstantVertex.of(priorOnInterceptScaleParameter));
        }
        public DoubleVertex getInterceptVertex(Double priorOnInterceptMean, Double priorOnInterceptScaleParameter) {
            return new GaussianVertex(priorOnInterceptMean, priorOnInterceptScaleParameter);
        }
    };

    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SCALE_PARAMETER = 2.0;

    public abstract DoubleVertex getWeightsVertex(long featureCount,  double[] priorOnWeightsMeans, double[] priorOnInterceptScaleParameter);
    public abstract DoubleVertex getInterceptVertex(Double priorOnInterceptMean, Double priorOnInterceptScaleParameter);
}
