package io.improbable.keanu.tensor;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public abstract class Nd4jTensor<T, TENSOR extends Tensor<T, TENSOR>> implements Tensor<T, TENSOR> {

    protected INDArray tensor;

    public Nd4jTensor(INDArray tensor) {
        this.tensor = tensor;
    }

    @Override
    public int getRank() {
        return tensor.rank();
    }

    @Override
    public long[] getShape() {
        return tensor.shape();
    }

    @Override
    public long[] getStride() {
        return tensor.stride();
    }

    @Override
    public long getLength() {
        return tensor.length();
    }

    @Override
    public TENSOR reshape(long... newShape) {
        return create(tensor.reshape(newShape));
    }

    @Override
    public TENSOR broadcast(long... toShape) {
        return create(tensor.broadcast(toShape));
    }

    @Override
    public TENSOR permute(int... rearrange) {
        return create(tensor.permute(rearrange));
    }

    @Override
    public TENSOR diag() {
        return create(Nd4j.diag(tensor));
    }

    @Override
    public TENSOR duplicate() {
        return create(tensor.dup());
    }

    @Override
    public TENSOR transpose() {
        if (this.getRank() != 2) {
            throw new IllegalArgumentException("Cannot transpose rank " + this.getRank() + " tensor. Try permute instead.");
        }
        return create(tensor.transpose());
    }

    @Override
    public TENSOR slice(int dimension, long index) {
        return create(tensor.slice(index, dimension));
    }

    @Override
    public TENSOR take(long... index) {
        return create(tensor.getScalar(index));
    }

    @Override
    public List<TENSOR> split(int dimension, long... splitAtIndices) {

        List<INDArray> splitINDArrays = INDArrayExtensions.split(tensor, dimension, splitAtIndices);

        return splitINDArrays.stream()
            .map(this::create)
            .collect(Collectors.toList());
    }

    @Override
    public int hashCode() {
        return tensor.hashCode();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;

        if (o instanceof Nd4jTensor) {
            return tensor.equals(((Nd4jTensor) o).getTensor());
        } else if (o instanceof Tensor) {
            Tensor that = (Tensor) o;
            if (!Arrays.equals(that.getShape(), getShape())) return false;
            return Arrays.equals(
                that.asFlatArray(),
                this.asFlatArray()
            );
        }

        return false;
    }

    @Override
    public String toString() {
        return tensor.toString();
    }

    public INDArray getTensor() {
        return tensor;
    }

    protected abstract INDArray getTensor(Tensor<T, ?> tensor);

    protected abstract TENSOR create(INDArray tensor);

    protected abstract TENSOR set(INDArray tensor);

    protected abstract TENSOR getThis();
}
