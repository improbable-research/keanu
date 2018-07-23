package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import java.util.NoSuchElementException;

import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.MissingParameterException;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.BinomialVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;

public class DistributionVertexBuilder {

    private int[] shape;
    private ParameterMap<Vertex<?>> parameters = new ParameterMap<>();

    public DistributionVertexBuilder shaped(int... shape) {
        this.shape = shape;
        return this;
    }

    public DistributionVertexBuilder withInput(ParameterName name, Double input) {
        return withInput(name, ConstantVertex.of(input));
    }
    public DistributionVertexBuilder withInput(ParameterName name, DoubleVertex input) {
        parameters.put(name, input);
        return this;
    }
    public DistributionVertexBuilder withInput(ParameterName name, Integer input) {
        return withInput(name, ConstantVertex.of(input));
    }
    public DistributionVertexBuilder withInput(ParameterName name, IntegerVertex input) {
        parameters.put(name, input);
        return this;
    }

    public BinomialVertex binomial() {
        try {
            Vertex<?> n = parameters.get(ParameterName.N).getValue();
            Vertex<?> p = parameters.get(ParameterName.P).getValue();
            if (shape == null) {
                shape = checkHasSingleNonScalarShapeOrAllScalar(p.getShape(), n.getShape());
            }
            return new BinomialVertex(shape, (DoubleVertex) p, (IntegerVertex) n);
        } catch (NoSuchElementException | ClassCastException e) {
            throw new MissingParameterException("Missing one or more parameters");
        }
    }

    public PoissonVertex poisson() {
        try {
            Vertex<?> mu = parameters.get(ParameterName.MU).getValue();
            if (shape == null) {
                shape = mu.getShape();
            }
            return new PoissonVertex(shape, (DoubleVertex) mu);
        } catch (NoSuchElementException | ClassCastException e) {
            throw new MissingParameterException("Missing one or more parameters");
        }
    }

    public GaussianVertex gaussian() {
        try {
            Vertex<?> mu = parameters.get(ParameterName.MU).getValue();
            Vertex<?> sigma = parameters.get(ParameterName.SIGMA).getValue();
            if (shape == null) {
                shape = checkHasSingleNonScalarShapeOrAllScalar(mu.getShape(), sigma.getShape());
            }
            return new GaussianVertex(shape, (DoubleVertex) mu, (DoubleVertex) sigma);
        } catch (NoSuchElementException | ClassCastException e) {
            throw new MissingParameterException("Missing one or more parameters");
        }
    }

    public InverseGammaVertex inverseGamma() {
        try {
            Vertex<?> param1 = parameters.get(ParameterName.A).getValue();
            Vertex<?> param2 = parameters.get(ParameterName.B).getValue();
            if (shape == null) {
                shape = checkHasSingleNonScalarShapeOrAllScalar(param1.getShape(), param2.getShape());
            }
            return new InverseGammaVertex(shape, (DoubleVertex) param1, (DoubleVertex) param2);
        } catch (NoSuchElementException | ClassCastException e) {
            throw new MissingParameterException("Missing one or more parameters");
        }
    }
}
