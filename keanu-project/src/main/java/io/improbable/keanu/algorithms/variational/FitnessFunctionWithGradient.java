package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.vertices.Vertex;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FitnessFunctionWithGradient extends FitnessFunction {

    public FitnessFunctionWithGradient(List<Vertex<?>> fitnessVertices, List<? extends Vertex<Double>> latentVertices) {
        super(fitnessVertices, latentVertices);
    }

    public MultivariateVectorFunction gradient() {
        return point -> {

            setAndCascadePoint(point);

            Map<String, Double> diffs = getDiffsWithRespectToUpstreamLatents(probabilisticVertices);

            return alignGradientsToAppropriateIndex(diffs);
        };
    }

    protected void setPoint(double[] point) {
        for (int i = 0; i < point.length; i++) {
            latentVertices
                    .get(i)
                    .setValue(point[i]);
        }
    }

    /**
     * @param probabilisticVertices
     * @return
     */
    public Map<String, Double> getDiffsWithRespectToUpstreamLatents(List<Vertex<?>> probabilisticVertices) {
        final Map<String, Double> diffOfLogWrt = new HashMap<>();

        for (final Vertex<?> probabilisticVertex : probabilisticVertices) {

            //Non-probabilistic vertices are non-differentiable
            if (!probabilisticVertex.isProbabilistic()) {
                continue;
            }

            //dlnDensityForProbabilisticVertex is the partial differentials of the natural
            //log of the fitness vertex's density w.r.t latent vertices. The key of the
            //map is the latent vertex's id.
            final Map<String, Double> dlnDensityForProbabilisticVertex = probabilisticVertex.dlnDensityAtValue();

            for (Map.Entry<String, Double> partialDiffLogPWrt : dlnDensityForProbabilisticVertex.entrySet()) {
                final String wrtLatentVertexId = partialDiffLogPWrt.getKey();
                final double partialDiffLogPContribution = partialDiffLogPWrt.getValue();

                //partialDiffLogPContribution is the contribution to the rate of change of
                //the natural log of the fitness vertex due to wrtLatentVertexId.
                final double accumulatedDiffOfLogPWrtLatent = diffOfLogWrt.getOrDefault(wrtLatentVertexId, 0.0);
                diffOfLogWrt.put(wrtLatentVertexId, accumulatedDiffOfLogPWrtLatent + partialDiffLogPContribution);
            }
        }

        return diffOfLogWrt;
    }

    private double[] alignGradientsToAppropriateIndex(Map<String /*Vertex Label*/, Double /*Gradient*/> diffs) {
        double[] gradient = new double[latentVertices.size()];
        for (int i = 0; i < gradient.length; i++) {
            gradient[i] = diffs.get(latentVertices.get(i).getId());
        }
        return gradient;
    }

}