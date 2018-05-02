package io.improbable.keanu.vertices.dbltensor;

import io.improbable.keanu.kotlin.DoubleOperators;
import org.nd4j.linalg.factory.Nd4j;

public interface DoubleTensor extends Tensor, DoubleOperators<DoubleTensor> {

    static DoubleTensor create(double[] values, int[] shape) {
        return new Nd4jDoubleTensor(values, shape);
    }

    static DoubleTensor ones(int[] shape) {
        return new Nd4jDoubleTensor(Nd4j.ones(shape));
    }

    static DoubleTensor zeros(int[] shape) {
        return new Nd4jDoubleTensor(Nd4j.zeros(shape));
    }

    double getValue(int[] index);

    void setValue(double value, int[] index);

    double sum();

    DoubleTensor reciprocal();

}
