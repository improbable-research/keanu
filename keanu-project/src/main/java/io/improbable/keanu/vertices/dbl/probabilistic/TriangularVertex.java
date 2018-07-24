package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class TriangularVertex extends DistributionBackedDoubleVertex<DoubleTensor> {

    /**
     * One xMin, xMax, c or all three that match a proposed tensor shape of Triangular
     *
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param xMin        the xMin of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param xMax        the xMax of the Triangular with either the same shape as specified for this vertex or a scalar
     * @param c           the center of the Triangular with either the same shape as specified for this vertex or a scalar
     */
    // package private
    TriangularVertex(int[] tensorShape, DoubleVertex xMin, DoubleVertex xMax, DoubleVertex c) {
        super(tensorShape, DistributionOfType::triangular, xMin, xMax, c);
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        throw new UnsupportedOperationException();
    }
}
