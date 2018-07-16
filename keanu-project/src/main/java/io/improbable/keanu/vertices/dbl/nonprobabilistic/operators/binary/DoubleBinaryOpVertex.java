package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;


import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;

public abstract class DoubleBinaryOpVertex extends NonProbabilisticDouble {

    protected final DoubleVertex a;
    protected final DoubleVertex b;

    /**
     * A vertex that performs a user defined operation on two vertices
     *
     * @param shape the shape of the resulting vertex
     * @param a a vertex
     * @param b a vertex
     */
    public DoubleBinaryOpVertex(int[] shape, DoubleVertex left, DoubleVertex right) {
        this.left = left;
        this.right = right;
        setParents(left, right);
        setValue(DoubleTensor.placeHolder(shape));
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(a.sample(random), b.sample(random));
    }

    @Override
    public DoubleTensor getDerivedValue() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract DoubleTensor op(DoubleTensor a, DoubleTensor b);

    public DoubleVertex getLeft(){
        return left;
    }

    public DoubleVertex getRight(){
        return right;
    }

}
