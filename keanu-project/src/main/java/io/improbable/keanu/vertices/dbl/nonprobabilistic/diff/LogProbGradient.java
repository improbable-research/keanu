package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;

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
    public static Map<String, Double> getJointLogProbGradientWrtLatents(List<Vertex> probabilisticVertices) {
        final Map<String, Double> diffOfLogWrt = new HashMap<>();

        for (final Vertex<?> probabilisticVertex : probabilisticVertices) {
            getLogProbGradientWrtLatents(probabilisticVertex, diffOfLogWrt);
        }

        return diffOfLogWrt;
    }

    public static Map<String, Double> getLogProbGradientWrtLatents(final Vertex<?> probabilisticVertex,
                                                                   final Map<String, Double> diffOfLogProbWrt) {
        //Non-probabilistic vertices are non-differentiable
        if (!probabilisticVertex.isProbabilistic()) {
            return diffOfLogProbWrt;
        }

        //dlogProbForProbabilisticVertex is the partial differentials of the natural
        //log of the fitness vertex's probability w.r.t latent vertices. The key of the
        //map is the latent vertex's id.
        final Map<String, DoubleTensor> dlogProbForProbabilisticVertex = probabilisticVertex.dLogProbAtValue();

        for (Map.Entry<String, DoubleTensor> partialDiffLogPWrt : dlogProbForProbabilisticVertex.entrySet()) {
            final String wrtLatentVertexId = partialDiffLogPWrt.getKey();
            final double partialDiffLogProbContribution = partialDiffLogPWrt.getValue().scalar();

            //partialDiffLogProbContribution is the contribution to the rate of change of
            //the natural log of the fitness vertex due to wrtLatentVertexId.
            final double accumulatedDiffOfLogPWrtLatent = diffOfLogProbWrt.getOrDefault(wrtLatentVertexId, 0.0);
            diffOfLogProbWrt.put(wrtLatentVertexId, accumulatedDiffOfLogPWrtLatent + partialDiffLogProbContribution);
        }

        return diffOfLogProbWrt;
    }

    public static Map<String, Double> getLogProbGradientWrtLatents(final Vertex<?> probabilisticVertex) {
        return getLogProbGradientWrtLatents(probabilisticVertex, new HashMap<>());
    }

}
