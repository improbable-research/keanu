package io.improbable.keanu.vertices.dbltensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

public class Nd4jDoubleTensor implements DoubleTensor {

    static {
        System.setProperty("dtype", "double");
    }

    public static final DoubleTensor ZERO_SCALAR = new Nd4jDoubleTensor(Nd4j.scalar(0.0));
    public static final DoubleTensor ONE_SCALAR = new Nd4jDoubleTensor(Nd4j.scalar(1.0));

    public static Nd4jDoubleTensor scalar(double scalarValue) {
        return new Nd4jDoubleTensor(Nd4j.scalar(scalarValue));
    }

    public static Nd4jDoubleTensor create(double[] values, int[] shape) {
        return new Nd4jDoubleTensor(values, shape);
    }

    public static Nd4jDoubleTensor ones(int[] shape) {
        return new Nd4jDoubleTensor(Nd4j.ones(shape));
    }

    public static Nd4jDoubleTensor zeros(int[] shape) {
        return new Nd4jDoubleTensor(Nd4j.zeros(shape));
    }

    private INDArray tensor;
    private int[] shape;

    public Nd4jDoubleTensor(double[] data, int[] shape) {
        tensor = Nd4j.create(data, shape);
        this.shape = shape;
    }

    public Nd4jDoubleTensor(int[] shape) {
        this.shape = shape;
    }

    public Nd4jDoubleTensor(INDArray tensor) {
        this.tensor = tensor;
        this.shape = tensor.shape();
    }

    @Override
    public int getRank() {
        return shape.length;
    }

    @Override
    public int[] getShape() {
        return shape;
    }

    @Override
    public int getLength() {
        return ArrayUtil.prod(shape);
    }

    public double getValue(int... index) {
        return tensor.getDouble(index);
    }

    public void setValue(double value, int... index) {
        tensor.putScalar(index, value);
    }

    public double sum() {
        return tensor.sumNumber().doubleValue();
    }

    @Override
    public double scalar() {
        return tensor.getDouble(0);
    }

    @Override
    public DoubleTensor reciprocal() {
        return new Nd4jDoubleTensor(tensor.rdiv(1.0));
    }

    @Override
    public DoubleTensor minus(double value) {
        return new Nd4jDoubleTensor(tensor.sub(value));
    }

    @Override
    public DoubleTensor plus(double value) {
        return new Nd4jDoubleTensor(tensor.add(value));
    }

    @Override
    public DoubleTensor times(double value) {
        return new Nd4jDoubleTensor(tensor.mul(value));
    }

    @Override
    public DoubleTensor div(double value) {
        return new Nd4jDoubleTensor(tensor.div(value));
    }

    @Override
    public DoubleTensor pow(DoubleTensor exponent) {
        if (exponent.isScalar()) {
            return pow(exponent.scalar());
        } else {
            INDArray exponentArray = unsafeGetNd4J(exponent);
            return new Nd4jDoubleTensor(Transforms.pow(tensor, exponentArray));
        }
    }

    @Override
    public DoubleTensor pow(double exponent) {
        return new Nd4jDoubleTensor(Transforms.pow(tensor, exponent));
    }

    @Override
    public DoubleTensor log() {
        return new Nd4jDoubleTensor(Transforms.log(tensor));
    }

    @Override
    public DoubleTensor sin() {
        return new Nd4jDoubleTensor(Transforms.sin(tensor));
    }

    @Override
    public DoubleTensor cos() {
        return new Nd4jDoubleTensor(Transforms.cos(tensor));
    }

    @Override
    public DoubleTensor asin() {
        return new Nd4jDoubleTensor(Transforms.asin(tensor));
    }

    @Override
    public DoubleTensor acos() {
        return new Nd4jDoubleTensor(Transforms.acos(tensor));
    }

    @Override
    public DoubleTensor exp() {
        return new Nd4jDoubleTensor(Transforms.exp(tensor));
    }

    @Override
    public DoubleTensor minus(DoubleTensor that) {

        if (that.isScalar() && !this.isScalar()) {
            return this.minus(that.scalar());
        } else if (!that.isScalar() && this.isScalar()) {
            return that.unaryMinus().plusInPlace(this);
        } else {
            INDArray thatArray = unsafeGetNd4J(that);
            return new Nd4jDoubleTensor(tensor.sub(thatArray));
        }
    }

    @Override
    public DoubleTensor plus(DoubleTensor that) {

        if (that.isScalar() && !this.isScalar()) {
            return this.plus(that.scalar());
        } else if (!that.isScalar() && this.isScalar()) {
            return that.plus(this.scalar());
        } else {
            INDArray thatArray = unsafeGetNd4J(that);
            return new Nd4jDoubleTensor(tensor.add(thatArray));
        }
    }

    @Override
    public DoubleTensor times(DoubleTensor that) {

        if (that.isScalar() && !this.isScalar()) {
            return this.times(that.scalar());
        } else if (!that.isScalar() && this.isScalar()) {
            return that.times(this.scalar());
        } else {
            INDArray thatArray = unsafeGetNd4J(that);
            return new Nd4jDoubleTensor(tensor.mul(thatArray));
        }
    }

    @Override
    public DoubleTensor div(DoubleTensor that) {

        if (that.isScalar() && !this.isScalar()) {
            return this.div(that.scalar());
        } else if (!that.isScalar() && this.isScalar()) {
            return that.reciprocal().timesInPlace(this);
        } else {
            INDArray thatArray = unsafeGetNd4J(that);
            return new Nd4jDoubleTensor(tensor.div(thatArray));
        }
    }

    @Override
    public DoubleTensor unaryMinus() {
        return new Nd4jDoubleTensor(tensor.neg());
    }

    @Override
    public DoubleTensor reciprocalInPlace() {
        tensor.rdivi(1.0);
        return this;
    }

    @Override
    public DoubleTensor minusInPlace(double value) {
        tensor.subi(value);
        return this;
    }

    @Override
    public DoubleTensor plusInPlace(double value) {
        tensor.addi(value);
        return this;
    }

    @Override
    public DoubleTensor timesInPlace(double value) {
        tensor.muli(value);
        return this;
    }

    @Override
    public DoubleTensor divInPlace(double value) {
        tensor.divi(value);
        return this;
    }

    @Override
    public DoubleTensor powInPlace(DoubleTensor exponent) {
        if (exponent.isScalar()) {
            Transforms.pow(tensor, exponent.scalar(), false);
        } else {
            INDArray exponentArray = unsafeGetNd4J(exponent);
            Transforms.pow(tensor, exponentArray, false);
        }
        return this;
    }

    @Override
    public DoubleTensor powInPlace(double exponent) {
        Transforms.pow(tensor, exponent, false);
        return this;
    }

    @Override
    public DoubleTensor logInPlace() {
        Transforms.log(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor sinInPlace() {
        Transforms.sin(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor cosInPlace() {
        Transforms.cos(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor asinInPlace() {
        Transforms.asin(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor acosInPlace() {
        Transforms.acos(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor expInPlace() {
        Transforms.exp(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor minusInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.subi(that.scalar());
        } else {
            INDArray thatArray = unsafeGetNd4J(that);
            tensor.subi(thatArray);
        }
        return this;
    }

    @Override
    public DoubleTensor plusInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.addi(that.scalar());
        } else {
            INDArray thatArray = unsafeGetNd4J(that);
            tensor.addi(thatArray);
        }
        return this;
    }

    @Override
    public DoubleTensor timesInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.muli(that.scalar());
        } else {
            INDArray thatArray = unsafeGetNd4J(that);
            tensor.muli(thatArray);
        }
        return this;
    }

    @Override
    public DoubleTensor divInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.divi(that.scalar());
        } else {
            INDArray thatArray = unsafeGetNd4J(that);
            tensor.divi(thatArray);
        }
        return this;
    }

    @Override
    public DoubleTensor unaryMinusInPlace() {
        tensor.negi();
        return this;
    }

    private INDArray unsafeGetNd4J(DoubleTensor that) {
        return ((Nd4jDoubleTensor) that).tensor;
    }

    @Override
    public double[] getLinearView() {
        return tensor.linearView().toDoubleVector();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Nd4jDoubleTensor that = (Nd4jDoubleTensor) o;

        return tensor.equals(that.tensor);
    }

    @Override
    public int hashCode() {
        return tensor.hashCode();
    }

    @Override
    public String toString() {
        return tensor.toString();
    }
}
