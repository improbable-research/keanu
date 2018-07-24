package io.improbable.keanu.vertices.intgr.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public abstract class DistributionBackedIntegerVertex<T extends Tensor<?>>
    extends IntegerVertex implements Probabilistic<IntegerTensor> {
    private final Function<List<T>, DiscreteDistribution> distributionCreator;

    public DistributionBackedIntegerVertex(int[] tensorShape, Function<List<T>, DiscreteDistribution> distributionCreator, Vertex<? extends T>... parents) {
        super(new ProbabilisticValueUpdater<>(), Observable.observableTypeFor(DistributionBackedIntegerVertex.class));
        this.distributionCreator = distributionCreator;

        List<TensorShape> inputShapes = ImmutableList.copyOf(parents)
            .stream()
            .map(v -> v.getShape())
            .map(TensorShape::new)
            .collect(Collectors.toList());

        checkTensorsMatchNonScalarShapeOrAreScalar(
            tensorShape,
            inputShapes
        );

        setParents(parents);
        setValue(IntegerTensor.placeHolder(tensorShape));
    }

    @Override
    public double logProb(IntegerTensor value) {
        return distribution().logProb(value).sum();
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return distribution().sample(getShape(), random);
    }

    protected DiscreteDistribution distribution() {
        return distributionCreator.apply(getInputs());
    }

    protected List<T> getInputs() {
        return getParents().stream().map(v -> ((Vertex<? extends T>)v).getValue()).collect(Collectors.toList());
    }
}
