package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public abstract class DistributionBackedDoubleVertex<T extends Tensor<?>>
    extends DoubleVertex implements Probabilistic<DoubleTensor> {
    private final Function<List<T>, ContinuousDistribution> distributionCreator;

    public  DistributionBackedDoubleVertex(int[] tensorShape, Function<List<T>, ContinuousDistribution> distributionCreator, Vertex<? extends T>... parents) {
        super(new ProbabilisticValueUpdater<>(), Observable.observableTypeFor(DistributionBackedDoubleVertex.class));
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
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    @Override
    public double logProb(DoubleTensor value) {
        return distribution().logProb(value).sum();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return distribution().sample(getShape(), random);
    }

    protected ContinuousDistribution distribution() {
        return distributionCreator.apply(getInputs());
    }

    protected List<T> getInputs() {
        return getParents().stream().map(v -> ((Vertex<? extends T>)v).getValue()).collect(Collectors.toList());
    }
}
