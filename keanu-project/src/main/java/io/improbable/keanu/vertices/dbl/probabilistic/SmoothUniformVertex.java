package io.improbable.keanu.vertices.dbl.probabilistic;

import static java.util.Collections.singletonMap;

import static io.improbable.keanu.distributions.dual.ParameterName.X;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class SmoothUniformVertex extends DistributionBackedDoubleVertex<DoubleTensor> {

    /**
     * One xMin or Xmax or both that match a proposed tensor shape of Smooth Uniform
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape   the desired shape of the vertex
     * @param xMin          the xMin of the Smooth Uniform with either the same shape as specified for this vertex or a scalar
     * @param xMax          the xMax of the Smooth Uniform with either the same shape as specified for this vertex or a scalar
     * @param edgeSharpness the edge sharpness of the Smooth Uniform
     */
    // package private
    SmoothUniformVertex(int[] tensorShape, DoubleVertex xMin, DoubleVertex xMax, DoubleVertex edgeSharpness) {
        super(tensorShape, DistributionOfType::smoothUniform, xMin, xMax, edgeSharpness);
    }

    @Override
    public double logProb(DoubleTensor value) {
        final DoubleTensor density = distribution().logProb(value);
        return density.logInPlace().sum();
    }

        @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        final DoubleTensor dPdx = distribution().dLogProb(value).get(X).getValue();
        final DoubleTensor density = distribution().logProb(value);
        final DoubleTensor dLogPdx = dPdx.divInPlace(density);
        return singletonMap(getId(), dLogPdx);
    }
}
