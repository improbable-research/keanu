package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.8 page 48
 */
public class TensorUniform {

    private TensorUniform() {
    }

    /**
     * @param xMin   minimum x value
     * @param xMax   maximum x value
     * @param random source of randomness
     * @return a random number from the Uniform distribution
     */
    public static DoubleTensor sample(DoubleTensor xMin, DoubleTensor xMax, KeanuRandom random) {
        return random.nextDouble(xMax.getShape()).timesInPlace(xMax.minus(xMin)).plusInPlace(xMin);
    }

    public static DoubleTensor logPdf(DoubleTensor xMin, DoubleTensor xMax, DoubleTensor x) {

        DoubleTensor logOfWithinBounds = xMax.minus(xMin).logInPlace().unaryMinusInPlace();


        //TODO: zero out where out of bounds
        return logOfWithinBounds;
    }
}
