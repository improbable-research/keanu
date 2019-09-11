package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Dirichlet;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.distributions.hyperparam.Diffs.C;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;

public class DirichletVertex extends VertexImpl<DoubleTensor, DoubleVertex>
    implements ProbabilisticDouble, Differentiable, LogProbGraphSupplier {

    private final DoubleVertex concentration;

    private static final String CONCENTRATION_NAME = "concentration";

    /**
     * Dirichlet distribution. The shape is driven from concentration, which must be a vector.
     *
     * @param tensorShape   the desired shape of the vertex
     * @param concentration the concentration values of the dirichlet
     */
    public DirichletVertex(@LoadShape long[] tensorShape,
                           @LoadVertexParam(CONCENTRATION_NAME) Vertex<DoubleTensor, ?> concentration) {
        super(tensorShape);
        this.concentration = wrapIfNeeded(concentration);
        if (concentration.getValue().getLength() < 2) {
            throw new IllegalArgumentException("Dirichlet must be comprised of more than one concentration parameter");
        }
        setParents(concentration);
    }

    /**
     * Matches a vector of concentration values to a Dirichlet distribution
     *
     * @param concentration the concentration values of the dirichlet
     */
    @ExportVertexToPythonBindings
    public DirichletVertex(Vertex<DoubleTensor, ?> concentration) {
        this(concentration.getShape(), concentration);
    }

    /**
     * Matches a scalar concentration value to a desired shape of a Dirichlet distribution
     *
     * @param tensorShape   the desired shape of the vertex
     * @param concentration the concentration values of the dirichlet
     */
    public DirichletVertex(long[] tensorShape, double concentration) {
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

    @SaveVertexParam(CONCENTRATION_NAME)
    public DoubleVertex getConcentration() {
        return concentration;
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor concentrationValues = concentration.getValue();
        DoubleTensor logPdfs = Dirichlet.withParameters(concentrationValues).logProb(value);
        return logPdfs.sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        DoublePlaceholderVertex xPlaceholder = new DoublePlaceholderVertex(this.getShape());
        DoublePlaceholderVertex concentrationPlaceholder = new DoublePlaceholderVertex(concentration.getShape());

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(concentration, concentrationPlaceholder)
            .logProbOutput(Dirichlet.logProbOutput(xPlaceholder, concentrationPlaceholder))
            .build();
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
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        return Dirichlet.withParameters(concentration.getValue()).sample(shape, random);
    }

    @Override
    public DoubleTensor upperBound() {
        return DoubleTensor.scalar(1.0);
    }

    @Override
    public DoubleTensor lowerBound() {
        return DoubleTensor.scalar(0.0);
    }
}
