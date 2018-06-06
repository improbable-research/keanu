package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

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
    public static Map<Long, DoubleTensor> getJointLogProbGradientWrtLatents(List<? extends Vertex> probabilisticVertices) {
        final Map<Long, DoubleTensor> diffOfLogWrt = new HashMap<>();

        for (final Vertex<?> probabilisticVertex : probabilisticVertices) {
            getLogProbGradientWrtLatents(probabilisticVertex, diffOfLogWrt);
        }

        return diffOfLogWrt;
    }

    public static Map<Long, DoubleTensor> getLogProbGradientWrtLatents(final Vertex<?> probabilisticVertex,
                                                                       final Map<Long, DoubleTensor> diffOfLogProbWrt) {
        //Non-probabilistic vertices are non-differentiable
        if (!probabilisticVertex.isProbabilistic()) {
            return diffOfLogProbWrt;
        }

        //dlogProbForProbabilisticVertex is the partial differentials of the natural
        //log of the fitness vertex's probability w.r.t latent vertices. The key of the
        //map is the latent vertex's id.
        final Map<Long, DoubleTensor> dlogProbForProbabilisticVertex = probabilisticVertex.dLogProbAtValue();

        for (Map.Entry<Long, DoubleTensor> partialDiffLogPWrt : dlogProbForProbabilisticVertex.entrySet()) {
            final long wrtLatentVertexId = partialDiffLogPWrt.getKey();
            final DoubleTensor partialDiffLogProbContribution = partialDiffLogPWrt.getValue();

            //partialDiffLogProbContribution is the contribution to the rate of change of
            //the natural log of the fitness vertex due to wrtLatentVertexId.
            final DoubleTensor accumulatedDiffOfLogPWrtLatent = diffOfLogProbWrt.get(wrtLatentVertexId);

            if (accumulatedDiffOfLogPWrtLatent == null) {
                diffOfLogProbWrt.put(wrtLatentVertexId, partialDiffLogProbContribution);
            } else {
                diffOfLogProbWrt.put(wrtLatentVertexId, accumulatedDiffOfLogPWrtLatent.plus(partialDiffLogProbContribution));
            }
        }

        return diffOfLogProbWrt;
    }

    public static Map<Long, DoubleTensor> getLogProbGradientWrtLatents(final Vertex<?> probabilisticVertex) {
        return getLogProbGradientWrtLatents(probabilisticVertex, new HashMap<>());
    }

}
