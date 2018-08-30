package io.improbable.keanu.algorithms.variational;

import com.google.common.collect.Iterables;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDouble;

import java.util.function.Function;

public class KLDivergence {

    public static double compute(QDistribution q, NetworkSamples p) {
        return compute(p, q::getLogOfMasterP);
    }

    public static double compute(ProbabilisticDouble q, NetworkSamples p) {
        return compute(p, networkState -> q.logProb(networkState.get(Iterables.getOnlyElement(networkState.getVertexIds()))));
    }

    private static double compute(NetworkSamples samples, Function<NetworkState, Double> qLogProbCalculator) {
        double divergence = 0.;

        for (int i = 0; i < samples.size(); i++) {
            double pLogProb = samples.getLogOfMasterP(i);

            NetworkState state = samples.getNetworkState(i);
            double qLogProb = qLogProbCalculator.apply(state);

            if (pLogProb != Double.NEGATIVE_INFINITY) {
                if (qLogProb == Double.NEGATIVE_INFINITY) {
                    throw new IllegalArgumentException("Q cannot have smaller support than P");
                }
                divergence += (pLogProb - qLogProb) * Math.exp(pLogProb);
            }
        }

        return divergence;
    }
}