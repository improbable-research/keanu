package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;

public class LogProbGradientCalculator {

    private final List<? extends Vertex> ofVertices;
    private final List<? extends Vertex> wrtVertices;

    public LogProbGradientCalculator(List<? extends Vertex> ofVertices, List<? extends Vertex> wrtVertices) {
        this.ofVertices = ofVertices;
        this.wrtVertices = wrtVertices;
    }

    /**
     * @return the partial derivatives with respect to any latents upstream
     */
    public Map<VertexId, DoubleTensor> getJointLogProbGradientWrtLatents() {
        final Map<VertexId, DoubleTensor> diffOfLogWrt = new HashMap<>();

        for (final Vertex<?> probabilisticVertex : ofVertices) {
            getLogProbGradientWrtLatents(probabilisticVertex, diffOfLogWrt);
        }

        return diffOfLogWrt;
    }

    public <T> Map<VertexId, DoubleTensor> getLogProbGradientWrtLatents(final Vertex<T> probabilisticVertex,
                                                                        final Map<VertexId, DoubleTensor> diffOfLogProbWrt) {
        //dlogProbForProbabilisticVertex is the partial differentials of the natural
        //log of the fitness vertex's probability w.r.t latent vertices. The key of the
        //map is the latent vertex's id.
        final Map<VertexId, DoubleTensor> dlogProbForProbabilisticVertex = ((Probabilistic<T>) probabilisticVertex).dLogProbAtValue(probabilisticVertex.getParents());

        for (Map.Entry<VertexId, DoubleTensor> partialDiffLogPWrt : dlogProbForProbabilisticVertex.entrySet()) {
            final VertexId wrtLatentVertexId = partialDiffLogPWrt.getKey();
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

}
