package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.BuilderParameterException;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.MissingParameterException;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.BinomialVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;

public class DistributionVertexBuilder {
    private static Logger log = LoggerFactory.getLogger(DistributionVertexBuilder.class);

    private int[] shape;
    private ParameterMap<Vertex<?>> parameters = new ParameterMap<>();

    public DistributionVertexBuilder shaped(int... shape) {
        this.shape = shape;
        return this;
    }

    public DistributionVertexBuilder withInput(ParameterName name, Double input) {
        return withInput(name, ConstantVertex.of(input));
    }

    public DistributionVertexBuilder withInput(ParameterName name, DoubleTensor input) {
        return withInput(name, ConstantVertex.of(input));
    }

    public DistributionVertexBuilder withInput(ParameterName name, DoubleVertex input) {
        parameters.put(name, input);
        return this;
    }

    public DistributionVertexBuilder withInput(ParameterName name, Integer input) {
        return withInput(name, ConstantVertex.of(input));
    }

    public DistributionVertexBuilder withInput(ParameterName name, IntegerTensor input) {
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
        return build(GaussianVertex.class, ParameterName.MU, ParameterName.SIGMA);
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


    public StudentTVertex studentT() {
        return build(StudentTVertex.class, ParameterName.V);
    }

    public UniformVertex uniform() {
        return build(UniformVertex.class, ParameterName.MIN, ParameterName.MAX);
    }

    public UniformIntVertex uniformInt() {
        return build(UniformIntVertex.class, ParameterName.MIN, ParameterName.MAX);
    }

    private <V extends Vertex<? extends Tensor>> V build(Class<V> clazz, ParameterName... parameterNames) {
        if(parameterNames.length < parameters.size()) {
            throw new BuilderParameterException(String.format(
                "Too many input parameters - expected %s, got %s",
                parameterNames.length, parameters.size())
            );
        } else if (parameterNames.length > parameters.size()) {
            throw new MissingParameterException(String.format(
               "Not enough input parameters - expected %s, got %s",
               parameterNames.length, parameters.size()
            ));
        }
        try {
            List<? extends Vertex<?>> inputs = Arrays.stream(parameterNames)
                .map(name -> parameters.get(name).getValue())
                .collect(Collectors.toList());

            List<int[]> shapes = inputs.stream()
                .map(input -> ((Vertex) input).getShape())
                .collect(Collectors.toList());

            if (shape == null) {
                shape = checkHasSingleNonScalarShapeOrAllScalar(shapes.toArray(new int[0][0]));
            }
            return construct(clazz, inputs);
        } catch (IllegalArgumentException | IllegalAccessException | InvocationTargetException | InstantiationException e) {
            String message = String.format(
                "Failed to construct Distribution Vertex class %s with parameters %s",
                clazz.getSimpleName(), parameters);
            log.error(message, e);
            throw new IllegalArgumentException(message, e);

        } catch (NoSuchElementException | ClassCastException e) {
            throw new MissingParameterException("Missing one or more parameters");
        }
    }

    private <V extends Vertex<? extends Tensor>> V construct(Class<V> clazz, List<? extends Vertex<?>> inputs)
        throws IllegalAccessException, InvocationTargetException, InstantiationException {
        Constructor[] constructors = clazz.getDeclaredConstructors();
        Constructor constructor = constructors[0];
        List<Object> constructorArgs = ImmutableList.builder().add(shape).addAll(inputs).build();
        return (V) constructor.newInstance(constructorArgs.toArray());
    }
}
