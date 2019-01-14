package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.discrete.Bernoulli;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class BernoulliVertex extends BooleanVertex implements ProbabilisticBoolean, SamplableWithManyScalars<BooleanTensor> {

    private final DoubleVertex probTrue;
    private final static String PROBTRUE_NAME = "probTrue";

    /**
     * One probTrue that must match a proposed tensor shape of Bernoulli.
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param shape    the desired shape of the vertex
     * @param probTrue the probability the bernoulli returns true
     */
    public BernoulliVertex(@LoadShape long[] shape, @LoadVertexParam(PROBTRUE_NAME) DoubleVertex probTrue) {
        super(shape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(shape, probTrue.getShape());
        this.probTrue = probTrue;
        setParents(probTrue);
    }

    /**
     * One to one constructor for mapping some shape of probTrue to
     * a matching shaped Bernoulli.
     *
     * @param probTrue probTrue with same shape as desired Bernoulli tensor or scalar
     */
    @ExportVertexToPythonBindings
    public BernoulliVertex(DoubleVertex probTrue) {
        this(probTrue.getShape(), probTrue);
    }

    public BernoulliVertex(double probTrue) {
        this(Tensor.SCALAR_SHAPE, new ConstantDoubleVertex(probTrue));
    }

    public BernoulliVertex(long[] shape, double probTrue) {
        this(shape, new ConstantDoubleVertex(probTrue));
    }

    @SaveVertexParam(PROBTRUE_NAME)
    public DoubleVertex getProbTrue() {
        return probTrue;
    }

    @Override
    public double logProb(BooleanTensor value) {
        return Bernoulli.withParameters(probTrue.getValue()).logProb(value).sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(BooleanTensor value, Set<? extends Vertex> withRespectTo) {

        if (!(probTrue.isDifferentiable())) {
            throw new UnsupportedOperationException("The probability of the Bernoulli being true must be differentiable");
        }

        if (withRespectTo.contains(probTrue)) {
            DoubleTensor dLogPdp = Bernoulli.withParameters(probTrue.getValue()).dLogProb(value);
            return Collections.singletonMap(probTrue, dLogPdp);
        }

        return Collections.emptyMap();
    }

    @Override
    public BooleanTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Bernoulli.withParameters(probTrue.getValue()).sample(shape, random);
    }
}
