package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.model.MaximumLikelihoodModelFitter;
import io.improbable.keanu.model.ModelFitter;
import io.improbable.keanu.model.ModelGraph;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex;

public enum RegressionRegularization {
    NONE {
        public DoubleVertex getWeightsVertex(long featureCount,  DoubleVertex priorOnWeightsMeans, DoubleVertex priorOnWeightsScaleParameter) {
            return new GaussianVertex(new long[]{1, featureCount}, priorOnWeightsMeans, priorOnWeightsScaleParameter).setLabel("weights");
        }

        public DoubleVertex getInterceptVertex(DoubleVertex priorOnInterceptMean, DoubleVertex priorOnInterceptScaleParameter) {
            return new GaussianVertex(priorOnInterceptMean, priorOnInterceptScaleParameter).setLabel("intercept");
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

    public abstract DoubleVertex getWeightsVertex(long featureCount,  DoubleVertex priorOnWeightsMeans, DoubleVertex priorOnInterceptScaleParameter);

    public abstract DoubleVertex getInterceptVertex(DoubleVertex priorOnInterceptMean, DoubleVertex priorOnInterceptScaleParameter);

    public abstract <INPUT, OUTPUT> ModelFitter<INPUT, OUTPUT> createFitterForGraph(ModelGraph<INPUT, OUTPUT> graph);
}
