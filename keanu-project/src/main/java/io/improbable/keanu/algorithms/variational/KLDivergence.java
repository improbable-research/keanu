package io.improbable.keanu.algorithms.variational;

import com.google.common.collect.Iterables;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDouble;

import java.util.Set;
import java.util.function.Function;

public class KLDivergence {

    public static double compute(QDistribution q, NetworkSamples p) {
        return compute(p, q::getLogOfMasterP);
    }

    public static double compute(ProbabilisticDouble q, NetworkSamples p) {
        return compute(p, networkState -> {
            Set<VariableReference> variableReferences = networkState.getVariableReferences();
            if (variableReferences.size() != 1) {
                throw new IllegalArgumentException("A NetworkState does not contain exactly 1 variable and ProbabilisticDouble can only compute the log probability of one value. Try computing KL divergence against a QDistribution instead.");
            }

            return q.logProb(networkState.get(Iterables.getOnlyElement(variableReferences)));
        });
    }

    private static double compute(NetworkSamples samples, Function<NetworkState, Double> qLogProbCalculator) {
        double divergence = 0.;

        for (int i = 0; i < samples.size(); i++) {
            double pLogProb = samples.getLogOfMasterP(i);

            NetworkState state = samples.getNetworkState(i);
            double qLogProb = qLogProbCalculator.apply(state);

            if (!ProbabilityCalculator.isImpossibleLogProb(pLogProb)) {
                if (ProbabilityCalculator.isImpossibleLogProb(qLogProb)) {
                    throw new IllegalArgumentException("Q cannot have smaller support than P.");
                }
                divergence += (pLogProb - qLogProb) * Math.exp(pLogProb);
            }
        }

        return divergence;
    }
}