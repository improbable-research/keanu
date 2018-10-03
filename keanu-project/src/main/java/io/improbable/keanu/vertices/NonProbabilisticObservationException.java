package io.improbable.keanu.vertices;

public class NonProbabilisticObservationException extends RuntimeException {

    public static final String BAD_OBSERVATION_WARNING =
            "Observation of a non-probabilistic vertex"
                    + " has a significant negative impact on inference algorithms' ability to find probable"
                    + " states. Please connect your non-probabilistic vertex to a probabilistic vertex and"
                    + " then observe that vertex. (e.g. connect the non-probabilistic double vertex to a Gaussian"
                    + " vertex as the mu and then observe the Gaussian)";

    public NonProbabilisticObservationException() {
        super(BAD_OBSERVATION_WARNING);
    }
}
