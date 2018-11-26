package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.model.MaximumLikelihoodModelFitter;
import io.improbable.keanu.model.ModelFitter;
import io.improbable.keanu.model.ModelGraph;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex;

public enum RegressionRegularization {
    NONE {
        public DoubleVertex getWeightsVertex(long featureCount,  DoubleVertex priorOnWeightsMeans, DoubleVertex priorOnWeightsScaleParameter) {
            return new GaussianVertex(new long[]{1, featureCount}, DEFAULT_MU, DEFAULT_SCALE_PARAMETER);
        }
        public DoubleVertex getInterceptVertex(DoubleVertex priorOnInterceptMean, DoubleVertex priorOnInterceptScaleParameter) {
            return new GaussianVertex(DEFAULT_MU, DEFAULT_SCALE_PARAMETER);
        }
        public <INPUT, OUTPUT> ModelFitter<INPUT, OUTPUT> createFitterForGraph(ModelGraph<INPUT, OUTPUT> graph) {
            return new MaximumLikelihoodModelFitter<>(graph);
        }
    },
    LASSO {
        public DoubleVertex getWeightsVertex(long featureCount, DoubleVertex priorOnWeightsMeans, DoubleVertex priorOnInterceptScaleParameter) {
            return new LaplaceVertex(new long[]{1, featureCount}, priorOnWeightsMeans, priorOnInterceptScaleParameter);
        }
        public DoubleVertex getInterceptVertex(DoubleVertex priorOnInterceptMean, DoubleVertex priorOnInterceptScaleParameter) {
            return new LaplaceVertex(priorOnInterceptMean, priorOnInterceptScaleParameter);
        }
        public <INPUT, OUTPUT> ModelFitter<INPUT, OUTPUT> createFitterForGraph(ModelGraph<INPUT, OUTPUT> graph) {
            return new MAPModelFitter<>(graph);
        }
    },
    RIDGE {
        public DoubleVertex getWeightsVertex(long featureCount, DoubleVertex priorOnWeightsMeans, DoubleVertex priorOnWeightsScaleParameter) {
            return new GaussianVertex(new long[]{1, featureCount}, priorOnWeightsMeans, priorOnWeightsScaleParameter);
        }
        public DoubleVertex getInterceptVertex(DoubleVertex priorOnInterceptMean, DoubleVertex priorOnInterceptScaleParameter) {
            return new GaussianVertex(priorOnInterceptMean, priorOnInterceptScaleParameter);
        }
        public <INPUT, OUTPUT> ModelFitter<INPUT, OUTPUT> createFitterForGraph(ModelGraph<INPUT, OUTPUT> graph) {
            return new MAPModelFitter<>(graph);
        }
    };

    private static final ConstantDoubleVertex DEFAULT_MU = ConstantVertex.of(0.0);
    private static final ConstantDoubleVertex DEFAULT_SCALE_PARAMETER = ConstantVertex.of(2.0);

    public abstract DoubleVertex getWeightsVertex(long featureCount, DoubleVertex priorOnWeightsMeans, DoubleVertex priorOnWeightsScaleParameter);
    public abstract DoubleVertex getInterceptVertex(DoubleVertex priorOnInterceptMean, DoubleVertex priorOnInterceptScaleParameter);
    public abstract <INPUT, OUTPUT> ModelFitter<INPUT, OUTPUT> createFitterForGraph(ModelGraph<INPUT, OUTPUT> graph);
}
