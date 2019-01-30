package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.discrete.Geometric;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class GeometricVertex extends IntegerVertex implements ProbabilisticInteger, SamplableWithManyScalars<IntegerTensor> {

    private final DoubleVertex p;
    private final static String P_NAME = "p";

    /**
     * A Vertex representing a Geometrically distributed random variable.
     * <p>
     * The Keanu Implementation has a support of {1, 2, 3, ...} ie it produces the number of tests until success (not
     * the number of failures until success which has a support {0, 1, 2, ...}
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the tensor in this vertex
     * @param p           the probability that an individual test is a success
     */
    public GeometricVertex(@LoadShape long[] tensorShape,
                           @LoadVertexParam(P_NAME) DoubleVertex p) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, p.getShape());
        this.p = p;

        setParents(p);
    }

    public GeometricVertex(long[] tensorShape, double p) {
        this(tensorShape, ConstantVertex.of(p));
    }

    @ExportVertexToPythonBindings
    public GeometricVertex(DoubleVertex p) {
        this(p.getShape(), p);
    }

    public GeometricVertex(double p) {
        this(ConstantVertex.of(p));
    }

    @Override
    public IntegerTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Geometric.withParameters(p.getValue()).sample(shape, random);
    }

    @Override
    public double logProb(IntegerTensor value) {
        return Geometric.withParameters(p.getValue()).logProb(value).sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(IntegerTensor atValue, Set<? extends Vertex> withRespectTo) {
        return Collections.emptyMap();
    }

    @SaveVertexParam(P_NAME)
    public DoubleVertex getP() {
        return p;
    }
}
