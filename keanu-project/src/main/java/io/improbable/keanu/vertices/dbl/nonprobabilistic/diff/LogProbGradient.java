package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.vertices.Vertex;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LogProbGradient {

    /**
     * @param probabilisticVertices
     * @return the partial derivatives with respect to any latents upstream
     */
    public static Map<String, Double> getJointLogProbGradientWrtLatents(List<Vertex<?>> probabilisticVertices) {
        final Map<String, Double> diffOfLogWrt = new HashMap<>();

        for (final Vertex<?> probabilisticVertex : probabilisticVertices) {
            getLogProbGradientWrtLatents(probabilisticVertex, diffOfLogWrt);
        }

        return diffOfLogWrt;
    }

    public static Map<String, Double> getLogProbGradientWrtLatents(final Vertex<?> probabilisticVertex,
                                                                   final Map<String, Double> diffOfLogWrt) {
        //Non-probabilistic vertices are non-differentiable
        if (!probabilisticVertex.isProbabilistic()) {
            return diffOfLogWrt;
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

        return diffOfLogWrt;
    }

    public static Map<String, Double> getLogProbGradientWrtLatents(final Vertex<?> probabilisticVertex) {
        return getLogProbGradientWrtLatents(probabilisticVertex, new HashMap<>());
    }

}
