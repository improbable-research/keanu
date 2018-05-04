package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.8 page 48
 */
public class NDUniform {

    private NDUniform() {
    }

    /**
     * @param xMin   minimum x value
     * @param xMax   maximum x value
     * @param random source of randomness
     * @return a random number from the Uniform distribution
     */
    public static DoubleTensor sample(DoubleTensor xMin, DoubleTensor xMax, KeanuRandom random) {
        return random.nextDouble(xMax.getShape()).times(xMax.minus(xMin)).plus(xMin);
    }

    public static DoubleTensor pdf(DoubleTensor xMin, DoubleTensor xMax, DoubleTensor x) {

        DoubleTensor withinBounds = xMax.minus(xMin).reciprocalInPlace();

        //TODO: zero out where out of bounds
        return withinBounds;
    }
}
