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

    // withInput()

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

    // distributions

    public BetaVertex beta() {
        return build(BetaVertex.class, ParameterName.A, ParameterName.B);
    }

    public BinomialVertex binomial() {
        return build(BinomialVertex.class, ParameterName.P, ParameterName.N);
    }

    public ChiSquaredVertex chiSquared() {
        return build(ChiSquaredVertex.class, ParameterName.K);
    }

    public ExponentialVertex exponential() {
        return build(ExponentialVertex.class, ParameterName.LOCATION, ParameterName.LAMBDA);
    }

    public GammaVertex gamma() {
        return build(GammaVertex.class, ParameterName.LOCATION, ParameterName.THETA, ParameterName.K);
    }

    public GaussianVertex gaussian() {
        return build(GaussianVertex.class, ParameterName.MU, ParameterName.SIGMA);
    }

    public InverseGammaVertex inverseGamma() {
        return build(InverseGammaVertex.class, ParameterName.A, ParameterName.B);
    }

    public LaplaceVertex laplace() {
        return build(LaplaceVertex.class, ParameterName.MU, ParameterName.BETA);
    }

    public LogisticVertex logistic() {
        return build(LogisticVertex.class, ParameterName.MU, ParameterName.S);
    }

    public LogNormalVertex logNormal() {
        return build(LogNormalVertex.class, ParameterName.MU, ParameterName.SIGMA);
    }

    /**
     * NB this one is a special case.
     * MU and SIGMA have different tensor shapes
     * and the resulting tensor has the same shape as MU
     * @return new MultivariateGaussian object
     */
    public MultivariateGaussian multivariateGaussian() {
        int[] requiredShape = this.parameters.get(ParameterName.MU).getValue().getShape();
        if (shape != null && !Arrays.equals(shape, requiredShape)) {
            throw new BuilderParameterException(
                String.format("Shape %s does not match mu's shape %s",
                    Arrays.toString(shape), Arrays.toString(requiredShape)));
        }
        shape = requiredShape;
        return build(MultivariateGaussian.class, ParameterName.MU, ParameterName.SIGMA);
    }

    public PoissonVertex poisson() {
        return build(PoissonVertex.class, ParameterName.MU);
    }

    public SmoothUniformVertex smoothUniform() {
        return build(SmoothUniformVertex.class, ParameterName.MIN, ParameterName.MAX, ParameterName.SHARPNESS);
    }

    public StudentTVertex studentT() {
        return build(StudentTVertex.class, ParameterName.V);
    }

    public TriangularVertex triangular() {
        return build(TriangularVertex.class, ParameterName.MIN, ParameterName.MAX, ParameterName.C);
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
            throw new BuilderParameterException(message, e);

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
