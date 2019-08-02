package com.example;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Samplable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDouble;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class CustomProbabilisticVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, Samplable<DoubleTensor> {

    public CustomProbabilisticVertex() {
        //This sets the shape of the output. Looks like it's a vector of length 2.
        super(new long[]{2});
    }

    public double logProb(DoubleTensor value) {
        final double a = value.getValue(0);
        final double b = value.getValue(1);

        if (a < 0 || b < 0) {
            return Double.NEGATIVE_INFINITY;
        } else {
            return Math.log(Math.pow(a + b, -2.5));
        }
    }

    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor atValue, Set<? extends Vertex> withRespectTo) {
        //This only needs to be implemented if you are using gradient based methods to do inference
        final double a = atValue.getValue(0);
        final double b = atValue.getValue(1);

        final double dlogProbWrtAAndB;
        if (a < 0 || b < 0) {
            dlogProbWrtAAndB = 0.0;
        } else {
            dlogProbWrtAAndB = -2.5 / (a + b);
        }

        HashMap<Vertex, DoubleTensor> result = new HashMap<>();
        result.put(this, DoubleTensor.create(dlogProbWrtAAndB, dlogProbWrtAAndB));

        return result;
    }

    public DoubleTensor sample(KeanuRandom random) {
        //This needs to be implemented if you use the priors as a Metropolis hastings proposal
        throw new UnsupportedOperationException();
    }

}
