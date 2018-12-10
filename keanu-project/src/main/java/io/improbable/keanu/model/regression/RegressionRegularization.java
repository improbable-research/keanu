package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.model.MaximumLikelihoodModelFitter;
import io.improbable.keanu.model.ModelFitter;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex;

public enum RegressionRegularization {
    NONE {
        public DoubleVertex getWeightsVertex(long featureCount,  DoubleVertex priorOnWeightsMeans, DoubleVertex priorOnWeightsScaleParameter) {
            return new GaussianVertex(new long[]{featureCount, 1}, priorOnWeightsMeans, priorOnWeightsScaleParameter).setLabel("weights");
        }

        public DoubleVertex getInterceptVertex(DoubleVertex priorOnInterceptMean, DoubleVertex priorOnInterceptScaleParameter) {
            return new GaussianVertex(priorOnInterceptMean, priorOnInterceptScaleParameter).setLabel("intercept");
        }

        public ModelFitter createFitterForGraph() {
            return new MaximumLikelihoodModelFitter();
        }
    },
    LASSO {
        public DoubleVertex getWeightsVertex(long featureCount, DoubleVertex priorOnWeightsMeans, DoubleVertex priorOnWeightsScaleParameter) {
            return new LaplaceVertex(new long[]{featureCount, 1}, priorOnWeightsMeans, priorOnWeightsScaleParameter);
        }

        public DoubleVertex getInterceptVertex(DoubleVertex priorOnInterceptMean, DoubleVertex priorOnInterceptScaleParameter) {
            return new LaplaceVertex(priorOnInterceptMean, priorOnInterceptScaleParameter);
        }

        public ModelFitter createFitterForGraph() {
            return new MAPModelFitter();
        }
    },
    RIDGE {
        public DoubleVertex getWeightsVertex(long featureCount, DoubleVertex priorOnWeightsMeans, DoubleVertex priorOnWeightsScaleParameter) {
            return new GaussianVertex(new long[]{featureCount, 1}, priorOnWeightsMeans, priorOnWeightsScaleParameter);
        }

        public DoubleVertex getInterceptVertex(DoubleVertex priorOnInterceptMean, DoubleVertex priorOnInterceptScaleParameter) {
            return new GaussianVertex(priorOnInterceptMean, priorOnInterceptScaleParameter);
        }

        public ModelFitter createFitterForGraph() {
            return new MAPModelFitter();
        }
    };

    public abstract DoubleVertex getWeightsVertex(long featureCount,  DoubleVertex priorOnWeightsMeans, DoubleVertex priorOnWeightsScaleParameter);

    public abstract DoubleVertex getInterceptVertex(DoubleVertex priorOnInterceptMean, DoubleVertex priorOnInterceptScaleParameter);

    public abstract ModelFitter createFitterForGraph();
}
