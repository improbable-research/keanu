package io.improbable.keanu.vertices;

import io.improbable.keanu.KeanuRandom;

public interface Samplable<T> {
    /**
     * @param random source of randomness
     * @return a sample from the vertex's distribution. For non-probabilistic vertices,
     * this will always be the same value.
     */
    T sample(KeanuRandom random);

    default T sample() {
        return sample(KeanuRandom.getDefaultRandom());
    }

}
