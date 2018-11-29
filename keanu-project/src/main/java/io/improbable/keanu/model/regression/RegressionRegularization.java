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
        public DoubleVertex getWeightsVertex(long featureCount, double[] priorOnWeightsMeans, double[] priorOnInterceptScaleParameter) {
            return new GaussianVertex(new long[]{featureCount, 1},
                ConstantVertex.of(priorOnWeightsMeans, featureCount, 1),
                ConstantVertex.of(priorOnInterceptScaleParameter, featureCount, 1)).setLabel("weights");
        }

        public DoubleVertex getInterceptVertex(Double priorOnInterceptMean, Double priorOnInterceptScaleParameter) {
            return new GaussianVertex(priorOnInterceptMean, priorOnInterceptScaleParameter).setLabel("intercept");
        }

        public <INPUT, OUTPUT> ModelFitter<INPUT, OUTPUT> createFitterForGraph(ModelGraph<INPUT, OUTPUT> graph) {
            return new MaximumLikelihoodModelFitter<>(graph);
        }
    },
    LASSO {
        public DoubleVertex getWeightsVertex(long featureCount, double[] priorOnWeightsMeans, double[] priorOnInterceptScaleParameter) {
            return new LaplaceVertex(new long[]{featureCount, 1},
                new ConstantDoubleVertex(priorOnWeightsMeans, new long[]{priorOnWeightsMeans.length, 1}),
                new ConstantDoubleVertex(priorOnInterceptScaleParameter, new long[]{priorOnInterceptScaleParameter.length, 1})
            );
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
            return new GaussianVertex(new long[]{featureCount, 1},
                new ConstantDoubleVertex(priorOnWeightsMeans, new long[]{priorOnWeightsMeans.length, 1}),
                new ConstantDoubleVertex(priorOnInterceptScaleParameter, new long[]{priorOnInterceptScaleParameter.length, 1})
            );
        }

        public DoubleVertex getInterceptVertex(Double priorOnInterceptMean, Double priorOnInterceptScaleParameter) {
            return new GaussianVertex(priorOnInterceptMean, priorOnInterceptScaleParameter);
        }

        public <INPUT, OUTPUT> ModelFitter<INPUT, OUTPUT> createFitterForGraph(ModelGraph<INPUT, OUTPUT> graph) {
            return new MAPModelFitter<>(graph);
        }
    };

    public abstract DoubleVertex getWeightsVertex(long featureCount, double[] priorOnWeightsMeans, double[] priorOnInterceptScaleParameter);

    public abstract DoubleVertex getInterceptVertex(Double priorOnInterceptMean, Double priorOnInterceptScaleParameter);

    public abstract <INPUT, OUTPUT> ModelFitter<INPUT, OUTPUT> createFitterForGraph(ModelGraph<INPUT, OUTPUT> graph);
}
