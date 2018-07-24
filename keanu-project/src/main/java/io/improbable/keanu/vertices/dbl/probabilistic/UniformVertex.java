package io.improbable.keanu.vertices.dbl.probabilistic;

import static java.util.Collections.singletonMap;

import java.util.List;
import java.util.Map;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class UniformVertex extends DistributionBackedDoubleVertex<DoubleTensor> {

    /**
     * One xMin or xMax or both that match a proposed tensor shape of UniformInt Vertex
     *
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape desired tensor shape
     * @param xMin  the inclusive lower bound of the UniformInt with either the same shape as specified for this vertex or a scalar
     * @param xMax  the exclusive upper bound of the UniformInt with either the same shape as specified for this vertex or a scalar
     */
    public UniformVertex(int[] tensorShape, DoubleVertex xMin, DoubleVertex xMax) {
        super(tensorShape, DistributionOfType::uniform, xMin, xMax);
    }

    @Override
    public List<DoubleVertex> getParents() {
        return (List<DoubleVertex>) super.getParents();
    }

    public DoubleVertex getXMin() {
        return getParents().get(0);
    }

    public DoubleVertex getXMax() {
        return getParents().get(1);
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {

        DoubleTensor dlogPdf = DoubleTensor.zeros(this.getXMax().getShape());
        dlogPdf = dlogPdf.setWithMaskInPlace(value.getGreaterThanMask(getXMax().getValue()), Double.NEGATIVE_INFINITY);
        dlogPdf = dlogPdf.setWithMaskInPlace(value.getLessThanOrEqualToMask(getXMin().getValue()), Double.POSITIVE_INFINITY);

        return singletonMap(getId(), dlogPdf);
    }
}
