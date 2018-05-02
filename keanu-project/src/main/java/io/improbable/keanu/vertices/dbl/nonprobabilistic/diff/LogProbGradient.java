package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.vertices.ContinuousVertex;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LogProbGradient {

    private LogProbGradient() {
    }

    /**
     * @param probabilisticVertices vertices to use in LogProb calc
     * @return the partial derivatives with respect to any latents upstream
     */
    public static Map<String, Double> getJointLogProbGradientWrtLatents(List<? extends ContinuousVertex> probabilisticVertices) {
        final Map<String, Double> diffOfLogWrt = new HashMap<>();

        for (final ContinuousVertex<?> probabilisticVertex : probabilisticVertices) {
            getLogProbGradientWrtLatents(probabilisticVertex, diffOfLogWrt);
        }

        return diffOfLogWrt;
    }

    public static Map<String, Double> getLogProbGradientWrtLatents(final ContinuousVertex<?> probabilisticVertex,
                                                                   final Map<String, Double> diffOfLogProbWrt) {
        //Non-probabilistic vertices are non-differentiable
        if (!probabilisticVertex.isProbabilistic()) {
            return diffOfLogProbWrt;
        }

        //dlogProbForProbabilisticVertex is the partial differentials of the natural
        //log of the fitness vertex's probability w.r.t latent vertices. The key of the
        //map is the latent vertex's id.
        final Map<String, Double> dlogProbForProbabilisticVertex = probabilisticVertex.dLogProbAtValue();

        for (Map.Entry<String, Double> partialDiffLogPWrt : dlogProbForProbabilisticVertex.entrySet()) {
            final String wrtLatentVertexId = partialDiffLogPWrt.getKey();
            final double partialDiffLogProbContribution = partialDiffLogPWrt.getValue();

            //partialDiffLogProbContribution is the contribution to the rate of change of
            //the natural log of the fitness vertex due to wrtLatentVertexId.
            final double accumulatedDiffOfLogPWrtLatent = diffOfLogProbWrt.getOrDefault(wrtLatentVertexId, 0.0);
            diffOfLogProbWrt.put(wrtLatentVertexId, accumulatedDiffOfLogPWrtLatent + partialDiffLogProbContribution);
        }

        return diffOfLogProbWrt;
    }

    public static Map<String, Double> getLogProbGradientWrtLatents(final ContinuousVertex<?> probabilisticVertex) {
        return getLogProbGradientWrtLatents(probabilisticVertex, new HashMap<>());
    }

}
