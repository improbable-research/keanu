package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Logistic;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SamplableWithManyScalars;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.distributions.hyperparam.Diffs.MU;
import static io.improbable.keanu.distributions.hyperparam.Diffs.S;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class LogisticVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, SamplableWithManyScalars<DoubleTensor>, LogProbGraphSupplier {

    private final DoubleVertex mu;
    private final DoubleVertex s;
    private static final String MU_NAME = "mu";
    private static final String S_NAME = "s";

    /**
     * One mu or s or both driving an arbitrarily shaped tensor of Logistic
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param mu          the mu (location) of the Logistic with either the same shape as specified for this vertex or mu scalar
     * @param s           the s (scale) of the Logistic with either the same shape as specified for this vertex or mu scalar
     */
    public LogisticVertex(@LoadShape long[] tensorShape,
                          @LoadVertexParam(MU_NAME) DoubleVertex mu,
                          @LoadVertexParam(S_NAME) DoubleVertex s) {
        super(tensorShape);
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(tensorShape, mu.getShape(), s.getShape());

        this.mu = mu;
        this.s = s;
        setParents(mu, s);
    }

    @ExportVertexToPythonBindings
    public LogisticVertex(DoubleVertex mu, DoubleVertex s) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(mu.getShape(), s.getShape()), mu, s);
    }

    public LogisticVertex(DoubleVertex mu, double s) {
        this(mu, new ConstantDoubleVertex(s));
    }

    public LogisticVertex(double mu, DoubleVertex s) {
        this(new ConstantDoubleVertex(mu), s);
    }

    public LogisticVertex(double mu, double s) {
        this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(s));
    }

    @SaveVertexParam(MU_NAME)
    public DoubleVertex getMu() {
        return mu;
    }

    @SaveVertexParam(S_NAME)
    public DoubleVertex getS() {
        return s;
    }

    @Override
    public double logProb(DoubleTensor value) {
        DoubleTensor muValues = mu.getValue();
        DoubleTensor sValues = s.getValue();

        DoubleTensor logPdfs = Logistic.withParameters(muValues, sValues).logProb(value);

        return logPdfs.sum();
    }

    @Override
    public LogProbGraph logProbGraph() {
        final LogProbGraph.DoublePlaceholderVertex xPlaceholder = new LogProbGraph.DoublePlaceholderVertex(this.getShape());
        final LogProbGraph.DoublePlaceholderVertex muPlaceholder = new LogProbGraph.DoublePlaceholderVertex(mu.getShape());
        final LogProbGraph.DoublePlaceholderVertex sPlaceholder = new LogProbGraph.DoublePlaceholderVertex(s.getShape());

        final DoubleVertex xMinusAOverB = xPlaceholder.minus(muPlaceholder).div(sPlaceholder);
        final DoubleVertex ln1OverB = ConstantVertex.of(1.).div(sPlaceholder).log();

        final DoubleVertex logProbOutput = xMinusAOverB.plus(ln1OverB).minus(xMinusAOverB.exp().plus(1).log().times(2));

        return LogProbGraph.builder()
            .input(this, xPlaceholder)
            .input(mu, muPlaceholder)
            .input(s, sPlaceholder)
            .logProbOutput(logProbOutput)
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Diffs dlnP = Logistic.withParameters(mu.getValue(), s.getValue()).dLogProb(value);

        Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

        if (withRespectTo.contains(mu)) {
            dLogProbWrtParameters.put(mu, dlnP.get(MU).getValue());
        }

        if (withRespectTo.contains(s)) {
            dLogProbWrtParameters.put(s, dlnP.get(S).getValue());
        }

        if (withRespectTo.contains(this)) {
            dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
        }

        return dLogProbWrtParameters;
    }

    @Override
    public DoubleTensor sampleWithShape(long[] shape, KeanuRandom random) {
        return Logistic.withParameters(mu.getValue(), s.getValue()).sample(shape, random);
    }
}
