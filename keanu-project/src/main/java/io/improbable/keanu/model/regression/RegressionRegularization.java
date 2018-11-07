package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.model.MaximumLikelihoodModelFitter;
import io.improbable.keanu.model.ModelFitter;
import io.improbable.keanu.model.ModelGraph;
import io.improbable.keanu.vertices.ConstantVertexFactory;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex;

public enum RegressionRegularization {
    NONE {
        public DoubleVertex getWeightsVertex(long featureCount,  double[] priorOnWeightsMeans, double[] priorOnInterceptScaleParameter) {
            return new GaussianVertex(new long[]{1, featureCount}, ConstantVertexFactory.of(DEFAULT_MU), ConstantVertexFactory.of(DEFAULT_SCALE_PARAMETER));
        }
        public DoubleVertex getInterceptVertex(Double priorOnInterceptMean, Double priorOnInterceptScaleParameter) {
            return new GaussianVertex(DEFAULT_MU, DEFAULT_SCALE_PARAMETER);
        }
        public <INPUT, OUTPUT> ModelFitter<INPUT, OUTPUT> createFitterForGraph(ModelGraph<INPUT, OUTPUT> graph) {
            return new MaximumLikelihoodModelFitter<>(graph);
        }
    },
    LASSO {
        public DoubleVertex getWeightsVertex(long featureCount, double[] priorOnWeightsMeans, double[] priorOnInterceptScaleParameter) {
            return new LaplaceVertex(new long[]{1, featureCount}, ConstantVertexFactory.of(priorOnWeightsMeans), ConstantVertexFactory.of(priorOnInterceptScaleParameter));
        }
        public DoubleVertex getInterceptVertex(Double priorOnInterceptMean, Double priorOnInterceptScaleParameter) {
            return new LaplaceVertex(priorOnInterceptMean, priorOnInterceptScaleParameter);
        }
        public <INPUT, OUTPUT> ModelFitter<INPUT, OUTPUT> createFitterForGraph(ModelGraph<INPUT, OUTPUT> graph) {
            return new MAPModelFitter<>(graph);
        }
    },
    RIDGE {
        public DoubleVertex getWeightsVertex(long featureCount, double[] priorOnWeightsMeans, double[] priorOnInterceptScaleParameter) {
            return new GaussianVertex(new long[]{1, featureCount}, ConstantVertexFactory.of(priorOnWeightsMeans), ConstantVertexFactory.of(priorOnInterceptScaleParameter));
        }
        public DoubleVertex getInterceptVertex(Double priorOnInterceptMean, Double priorOnInterceptScaleParameter) {
            return new GaussianVertex(priorOnInterceptMean, priorOnInterceptScaleParameter);
        }
        public <INPUT, OUTPUT> ModelFitter<INPUT, OUTPUT> createFitterForGraph(ModelGraph<INPUT, OUTPUT> graph) {
            return new MAPModelFitter<>(graph);
        }
    };

    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SCALE_PARAMETER = 2.0;

    public abstract DoubleVertex getWeightsVertex(long featureCount,  double[] priorOnWeightsMeans, double[] priorOnInterceptScaleParameter);
    public abstract DoubleVertex getInterceptVertex(Double priorOnInterceptMean, Double priorOnInterceptScaleParameter);
    public abstract <INPUT, OUTPUT> ModelFitter<INPUT, OUTPUT> createFitterForGraph(ModelGraph<INPUT, OUTPUT> graph);
}
