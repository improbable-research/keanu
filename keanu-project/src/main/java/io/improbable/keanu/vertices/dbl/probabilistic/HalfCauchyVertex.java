package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Cauchy;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.Map;
import java.util.Set;

public class HalfCauchyVertex extends CauchyVertex {

    private static final double LOC_ZERO = 0.0;
    private static final double LOG_TWO = Math.log(2);

    /**
     * One scale that matches a proposed tensor shape of HalfCauchy (Cauchy with location = 0 and non-negative x)
     * <p>
     * If provided parameter is scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param scale       the scale of the HalfCauchy with either the same tensorShape as specified for this vertex or a scalar
     */
    public HalfCauchyVertex(@LoadShape long[] tensorShape, @LoadVertexParam(SCALE_NAME) DoubleVertex scale) {
        super(tensorShape, LOC_ZERO, scale);
    }

    public HalfCauchyVertex(long[] tensorShape, double scale) {
        super(tensorShape, LOC_ZERO, scale);
    }

    @ExportVertexToPythonBindings
    public HalfCauchyVertex(DoubleVertex scale) {
        super(LOC_ZERO, scale);
    }

    public HalfCauchyVertex(double scale) {
        super(LOC_ZERO, scale);
    }

    @Override
    public double logProb(DoubleTensor value) {
        if (value.greaterThanOrEqual(LOC_ZERO).allTrue()) {
            return super.logProb(value) + LOG_TWO * value.getLength();
        }
        return Double.NEGATIVE_INFINITY;
    }

    @Override
    public LogProbGraph logProbGraph() {
        final DoublePlaceholderVertex xPlaceholder = new DoublePlaceholderVertex(this.getShape());
        final DoublePlaceholderVertex locationPlaceholder = new DoublePlaceholderVertex(getLocation().getShape());
        final DoublePlaceholderVertex scalePlaceholder = new DoublePlaceholderVertex(getScale().getShape());

        final DoubleVertex cauchyLogProbOutput = Cauchy.logProbOutput(xPlaceholder, locationPlaceholder, scalePlaceholder);

        final DoubleVertex result = cauchyLogProbOutput.plus(LOG_TWO);
        final DoubleVertex invalidMask = xPlaceholder.toLessThanMask(LOC_ZERO);
        final DoubleVertex halfCauchyLogProbOutput = result.setWithMask(invalidMask, Double.NEGATIVE_INFINITY);

        // Set the value of locationPlaceholder since we know it's 0.
        locationPlaceholder.setValue(LOC_ZERO);

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(getLocation(), locationPlaceholder)
            .input(getScale(), scalePlaceholder)
            .logProbOutput(halfCauchyLogProbOutput)
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Map<Vertex, DoubleTensor> logProb = super.dLogProb(value, withRespectTo);
        if (value.greaterThanOrEqual(LOC_ZERO).allTrue()) {
            return logProb;
        } else {
            for (Map.Entry<Vertex, DoubleTensor> entry : logProb.entrySet()) {
                DoubleTensor v = entry.getValue();
                logProb.put(entry.getKey(), v.setWithMaskInPlace(value.getLessThanMask(DoubleTensor.scalar(LOC_ZERO)), 0.0));
            }
            return logProb;
        }
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return super.sampleWithShape(shape, random).absInPlace();
    }

}
