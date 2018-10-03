package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.network.NetworkState;

/**
 * {@link QDistribution} represents Q in D(P|Q) = sum_i P(i) log(P(i)/Q(i)), which is the {@link
 * KLDivergence} (Kullback Leibler divergence) from probability distributions P to Q.
 */
public interface QDistribution {
    double getLogOfMasterP(NetworkState state);
}
