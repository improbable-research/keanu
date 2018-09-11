package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.Diffs.C;
import static io.improbable.keanu.distributions.dual.Diffs.X;
import static io.improbable.keanu.tensor.TensorShape.shapeToDesiredRankByPrependingOnes;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import io.improbable.keanu.distributions.continuous.Dirichlet;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class DirichletVertex extends DoubleVertex implements ProbabilisticDouble {

    private final DoubleVertex concentration;

    /**
     * Dirichlet distribution. The shape is driven from concentration, which must be a vector.
     *
     * @param tensorShape   the desired shape of the vertex
     * @param concentration the concentration values of the dirichlet
     */
    public DirichletVertex(int[] tensorShape, DoubleVertex concentration) {
        this.concentration = concentration;
        if (concentration.getValue().getLength() < 2) {
            throw new IllegalArgumentException("Dirichlet must be comprised of more than one concentration parameter");
        }
        setParents(concentration);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    /**
     * Matches a vector of concentration values to a Dirichlet distribution
     *
     * @param concentration the concentration values of the dirichlet
     */
    public DirichletVertex(DoubleVertex concentration) {
        this(concentration.getShape(), concentration);
    }

    /**
     * Matches a scalar concentration value to a desired shape of a Dirichlet distribution
     *
     * @param tensorShape   the desired shape of the vertex
     * @param concentration the concentration values of the dirichlet
     */
    public DirichletVertex(int[] tensorShape, double concentration) {
        this(tensorShape, new ConstantDoubleVertex(DoubleTensor.create(concentration, tensorShape)));
    }

    /**
     * Matches a vector of concentration values to a Dirichlet distribution
     *
     * @param concentration the concentration values of the dirichlet
     */
    public DirichletVertex(double... concentration) {
        this(new ConstantDoubleVertex(concentration));
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor concentrationValues = concentration.getValue();
        DoubleTensor logPdfs = Dirichlet.withParameters(concentrationValues).logProb(value);
        return logPdfs.sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Diffs dlnP = Dirichlet.withParameters(concentration.getValue()).dLogProb(value);

        Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

        if (withRespectTo.contains(concentration)) {
            dLogProbWrtParameters.put(concentration, dlnP.get(C).getValue());
        }

        if (withRespectTo.contains(this)) {
            dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
        }

        return dLogProbWrtParameters;

//        return convertDualNumbersToDiff(dlnP.get(C).getValue(), dlnP.get(X).getValue());
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Dirichlet.withParameters(concentration.getValue()).sample(getShape(), random);
    }

    private Map<VertexId, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdc,
                                                                 DoubleTensor dLogPdx) {

        PartialDerivatives dLogPdInputs = concentration.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdc);

        if (!this.isObserved()) {
            dLogPdInputs.putWithRespectTo(getId(), dLogPdx.reshape(
                shapeToDesiredRankByPrependingOnes(dLogPdx.getShape(), dLogPdx.getRank() + getValue().getRank()))
            );
        }

        PartialDerivatives summed = dLogPdInputs.sum(true, TensorShape.dimensionRange(0, getShape().length));

        return summed.asMap();
    }
}
