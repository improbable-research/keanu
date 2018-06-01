package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.SimpleBooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.intgr.Nd4jIntegerTensor;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.function.Function;

import static java.util.Arrays.copyOf;

public class Nd4jDoubleTensor implements DoubleTensor {

    static {
        System.setProperty("dtype", "double");
    }

    public static Nd4jDoubleTensor scalar(double scalarValue) {
        return new Nd4jDoubleTensor(Nd4j.scalar(scalarValue));
    }

    public static Nd4jDoubleTensor create(double[] values, int[] shape) {
        return new Nd4jDoubleTensor(values, shape);
    }

    public static Nd4jDoubleTensor create(double value, int[] shape) {
        return new Nd4jDoubleTensor(Nd4j.valueArrayOf(shape, value));
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
        DataBuffer buffer = Nd4j.createBuffer(data);
        tensor = Nd4j.create(buffer, shape);
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
    public long getLength() {
        return ArrayUtil.prodLong(shape);
    }

    @Override
    public boolean isShapePlaceholder() {
        return tensor == null;
    }

    public Double getValue(int... index) {
        return tensor.getDouble(index);
    }

    public void setValue(Double value, int... index) {
        tensor.putScalar(index, value);
    }

    public Double sum() {
        return tensor.sumNumber().doubleValue();
    }

    @Override
    public DoubleTensor duplicate() {
        return new Nd4jDoubleTensor(tensor.dup());
    }

    @Override
    public DoubleTensor apply(Function<Double, Double> function) {
        DataBuffer data = tensor.data().dup();
        for (int i = 0; i < data.length(); i++) {
            data.put(i, function.apply(data.getDouble(i)));
        }
        return new Nd4jDoubleTensor(data.asDouble(), this.getShape());
    }

    @Override
    public DoubleTensor max(DoubleTensor max) {
        return duplicate().maxInPlace(max);
    }

    @Override
    public DoubleTensor min(DoubleTensor min) {
        return duplicate().minInPlace(min);
    }

    @Override
    public DoubleTensor clamp(DoubleTensor min, DoubleTensor max) {
        return duplicate().clampInPlace(min, max);
    }

    @Override
    public DoubleTensor ceil() {
        return duplicate().ceilInPlace();
    }

    @Override
    public DoubleTensor floor() {
        return duplicate().floorInPlace();
    }

    @Override
    public DoubleTensor sigmoid() {
        return duplicate().sigmoidInPlace();
    }

    @Override
    public Double scalar() {
        return tensor.getDouble(0);
    }

    @Override
    public DoubleTensor reciprocal() {
        return duplicate().reciprocalInPlace();
    }

    @Override
    public DoubleTensor minus(double value) {
        return duplicate().minusInPlace(value);
    }

    @Override
    public DoubleTensor plus(double value) {
        return duplicate().plusInPlace(value);
    }

    @Override
    public DoubleTensor times(double value) {
        return duplicate().timesInPlace(value);
    }

    @Override
    public DoubleTensor div(double value) {
        return duplicate().divInPlace(value);
    }

    @Override
    public DoubleTensor pow(DoubleTensor exponent) {
        return duplicate().powInPlace(exponent);
    }

    @Override
    public DoubleTensor pow(double exponent) {
        return duplicate().powInPlace(exponent);
    }

    @Override
    public DoubleTensor sqrt() {
        return duplicate().sqrtInPlace();
    }

    @Override
    public DoubleTensor log() {
        return duplicate().logInPlace();
    }

    @Override
    public DoubleTensor sin() {
        return duplicate().sinInPlace();
    }

    @Override
    public DoubleTensor cos() {
        return duplicate().cosInPlace();
    }

    @Override
    public DoubleTensor tan() {
        return duplicate().tanInPlace();
    }

    @Override
    public DoubleTensor atan() {
        return duplicate().atanInPlace();
    }

    @Override
    public DoubleTensor atan2(double y) {
        return duplicate().atan2InPlace(y);
    }

    @Override
    public DoubleTensor atan2(DoubleTensor y) {
        return duplicate().atan2InPlace(y);
    }

    @Override
    public DoubleTensor asin() {
        return duplicate().asinInPlace();
    }

    @Override
    public DoubleTensor acos() {
        return duplicate().acosInPlace();
    }

    @Override
    public DoubleTensor exp() {
        return duplicate().expInPlace();
    }

    @Override
    public DoubleTensor minus(DoubleTensor that) {

        if (that.isScalar()) {
            return this.minus(that.scalar());
        } else if (this.isScalar()) {
            return that.unaryMinus().plusInPlace(this);
        } else {
            INDArray thatArray = unsafeGetNd4J(that);
            return new Nd4jDoubleTensor(tensor.sub(thatArray));
        }
    }

    @Override
    public DoubleTensor plus(DoubleTensor that) {

        if (that.isScalar()) {
            return this.plus(that.scalar());
        } else if (this.isScalar()) {
            return that.plus(this.scalar());
        } else {
            INDArray thatArray = unsafeGetNd4J(that);
            return new Nd4jDoubleTensor(tensor.add(thatArray));
        }
    }

    @Override
    public DoubleTensor times(DoubleTensor that) {

        if (that.isScalar()) {
            return this.times(that.scalar());
        } else if (this.isScalar()) {
            return that.times(this.scalar());
        } else {
            INDArray thatArray = unsafeGetNd4J(that);
            return new Nd4jDoubleTensor(tensor.mul(thatArray));
        }
    }

    @Override
    public DoubleTensor div(DoubleTensor that) {

        if (that.isScalar()) {
            return this.div(that.scalar());
        } else if (this.isScalar()) {
            return that.reciprocal().timesInPlace(this);
        } else {
            INDArray thatArray = unsafeGetNd4J(that);
            return new Nd4jDoubleTensor(tensor.div(thatArray));
        }
    }

    @Override
    public DoubleTensor abs() {
        return new Nd4jDoubleTensor(Transforms.abs(tensor));
    }

    @Override
    public DoubleTensor unaryMinus() {
        return duplicate().unaryMinusInPlace();
    }

    @Override
    public DoubleTensor setWithMask(DoubleTensor mask, double value) {
        return duplicate().setWithMaskInPlace(mask, value);
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
    public DoubleTensor sqrtInPlace() {
        Transforms.sqrt(tensor, false);
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
    public DoubleTensor tanInPlace() {
        INDArray sin = Transforms.sin(tensor, true);
        INDArray cos = Transforms.cos(tensor, true);
        tensor = sin.divi(cos);
        return this;
    }

    @Override
    public DoubleTensor atanInPlace() {
        Transforms.atan(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor atan2InPlace(double y) {
        return atan2InPlace(DoubleTensor.create(y, this.shape));
    }

    @Override
    public DoubleTensor atan2InPlace(DoubleTensor y) {
        if (y.isScalar()) {
            tensor = Transforms.atan2(tensor, Nd4j.valueArrayOf(this.shape, y.scalar()));
        }
        else {
            tensor = Transforms.atan2(tensor, unsafeGetNd4J(y));
        }
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

    @Override
    public DoubleTensor absInPlace() {
        return duplicate().absInPlace();
    }


    @Override
    public DoubleTensor getGreaterThanMask(DoubleTensor greaterThanThis) {

        INDArray mask = tensor.dup();

        if (greaterThanThis.isScalar()) {
            Nd4j.getExecutioner().exec(
                new OldGreaterThan(mask,
                    Nd4j.valueArrayOf(mask.shape(), greaterThanThis.scalar()),
                    mask,
                    mask.length()
                )
            );
        } else {
            INDArray greaterThanThisArray = unsafeGetNd4J(greaterThanThis);
            Nd4j.getExecutioner().exec(
                new OldGreaterThan(mask, greaterThanThisArray, mask, mask.length())
            );
        }

        return new Nd4jDoubleTensor(mask);
    }

    @Override
    public DoubleTensor getGreaterThanOrEqualToMask(DoubleTensor greaterThanOrEqualToThis) {

        INDArray mask = tensor.dup();

        if (greaterThanOrEqualToThis.isScalar()) {
            Nd4j.getExecutioner().exec(
                new OldGreaterThanOrEqual(mask,
                    Nd4j.valueArrayOf(mask.shape(), greaterThanOrEqualToThis.scalar()),
                    mask,
                    mask.length()
                )
            );
        } else {
            INDArray greaterThanThisArray = unsafeGetNd4J(greaterThanOrEqualToThis);
            Nd4j.getExecutioner().exec(
                new OldGreaterThanOrEqual(mask, greaterThanThisArray, mask, mask.length())
            );
        }

        return new Nd4jDoubleTensor(mask);
    }

    @Override
    public DoubleTensor getLessThanMask(DoubleTensor lessThanThis) {

        INDArray mask = tensor.dup();

        if (lessThanThis.isScalar()) {
            Nd4j.getExecutioner().exec(
                new OldLessThan(mask,
                    Nd4j.valueArrayOf(mask.shape(), lessThanThis.scalar()),
                    mask,
                    mask.length()
                )
            );
        } else {
            INDArray lessThanThisArray = unsafeGetNd4J(lessThanThis);
            Nd4j.getExecutioner().exec(
                new OldLessThan(mask, lessThanThisArray, mask, mask.length())
            );
        }

        return new Nd4jDoubleTensor(mask);
    }

    @Override
    public DoubleTensor getLessThanOrEqualToMask(DoubleTensor lessThanOrEqualToThis) {

        INDArray mask = tensor.dup();

        if (lessThanOrEqualToThis.isScalar()) {
            Nd4j.getExecutioner().exec(
                new OldLessThanOrEqual(mask,
                    Nd4j.valueArrayOf(mask.shape(), lessThanOrEqualToThis.scalar()),
                    mask,
                    mask.length()
                )
            );
        } else {
            INDArray lessThanOrEqualToThisArray = unsafeGetNd4J(lessThanOrEqualToThis);
            Nd4j.getExecutioner().exec(
                new OldLessThanOrEqual(mask, lessThanOrEqualToThisArray, mask, mask.length())
            );
        }

        return new Nd4jDoubleTensor(mask);
    }

    @Override
    public DoubleTensor setWithMaskInPlace(DoubleTensor mask, double value) {

        INDArray maskDup = unsafeGetNd4J(mask).dup();

        if (value == 0.0) {
            INDArray swapOnesForZeros = Nd4j.ones(tensor.shape()).subi(maskDup);
            tensor.muli(swapOnesForZeros);
        } else {
            Nd4j.getExecutioner().exec(
                new CompareAndSet(maskDup, value, Conditions.equals(1.0))
            );

            Nd4j.getExecutioner().exec(
                new CompareAndSet(tensor, maskDup, Conditions.notEquals(0.0))
            );
        }

        return this;
    }

    @Override
    public DoubleTensor applyInPlace(Function<Double, Double> function) {
        DataBuffer data = tensor.data();
        for (int i = 0; i < data.length(); i++) {
            data.put(i, function.apply(data.getDouble(i)));
        }
        return this;
    }

    @Override
    public DoubleTensor maxInPlace(DoubleTensor max) {
        if (max.isScalar()) {
            Transforms.max(tensor, max.scalar(), false);
        } else {
            Transforms.max(tensor, unsafeGetNd4J(max), false);
        }
        return this;
    }

    @Override
    public DoubleTensor minInPlace(DoubleTensor max) {
        if (max.isScalar()) {
            Transforms.min(tensor, max.scalar(), false);
        } else {
            Transforms.min(tensor, unsafeGetNd4J(max), false);
        }
        return this;
    }

    @Override
    public DoubleTensor clampInPlace(DoubleTensor min, DoubleTensor max) {
        return minInPlace(min).maxInPlace(max);
    }

    @Override
    public DoubleTensor ceilInPlace() {
        Transforms.ceil(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor floorInPlace() {
        Transforms.floor(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor sigmoidInPlace() {
        Transforms.sigmoid(tensor, false);
        return this;
    }

    // Comparisons

    @Override
    public BooleanTensor lessThan(double value) {
        return fromMask(tensor.lt(value), copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor lessThanOrEqual(double value) {
        return fromMask(tensor.lte(value), copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor lessThan(DoubleTensor value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.lt(value.scalar());
        } else {
            INDArray indArray = unsafeGetNd4J(value);
            mask = tensor.lt(indArray);
        }

        return fromMask(mask, copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor lessThanOrEqual(DoubleTensor value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.lte(value.scalar());
        } else {

            INDArray indArray = unsafeGetNd4J(value);
            mask = tensor.dup();
            Nd4j.getExecutioner().exec(new OldLessThanOrEqual(mask, indArray, mask, getLength()));
        }

        return fromMask(mask, copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor greaterThan(double value) {
        return fromMask(tensor.gt(value), copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor greaterThanOrEqual(double value) {
        return fromMask(tensor.gte(value), copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor greaterThan(DoubleTensor value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.gt(value.scalar());
        } else {
            INDArray indArray = unsafeGetNd4J(value);
            mask = tensor.gt(indArray);
        }

        return fromMask(mask, copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor greaterThanOrEqual(DoubleTensor value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.gt(value.scalar());
        } else {
            INDArray indArray = unsafeGetNd4J(value);
            mask = tensor.dup();
            Nd4j.getExecutioner().exec(new OldGreaterThanOrEqual(mask, indArray, mask, getLength()));
        }

        return fromMask(mask, copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor elementwiseEquals(Tensor that) {

        if (that instanceof Nd4jDoubleTensor) {
            INDArray eq = tensor.eq(unsafeGetNd4J((Nd4jDoubleTensor) that));
            return fromMask(eq, shape);
        } else {
            return Tensor.elementwiseEquals(this, that);
        }
    }

    private INDArray unsafeGetNd4J(DoubleTensor that) {
        return ((Nd4jDoubleTensor) that).tensor;
    }

    @Override
    public FlattenedView<Double> getFlattenedView() {
        return new Nd4jDoubleFlattenedView(tensor);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;

        if (o instanceof Nd4jDoubleTensor) {
            return tensor.equals(((Nd4jDoubleTensor) o).tensor);
        } else if (o instanceof Tensor) {
            Tensor that = (Tensor) o;
            if (!Arrays.equals(that.getShape(), shape)) return false;
            return Arrays.equals(
                that.asFlatArray(),
                this.asFlatArray()
            );
        }

        return false;
    }

    @Override
    public int hashCode() {
        return tensor.hashCode();
    }

    @Override
    public String toString() {
        return tensor.toString();
    }

    @Override
    public DoubleTensor toDouble() {
        return this;
    }

    @Override
    public IntegerTensor toInteger() {
        Transforms.floor(tensor, false);
        return new Nd4jIntegerTensor(tensor);
    }

    private BooleanTensor fromMask(INDArray mask, int[] shape) {
        DataBuffer data = mask.data();
        boolean[] boolsFromMask = new boolean[mask.length()];

        for (int i = 0; i < boolsFromMask.length; i++) {
            boolsFromMask[i] = data.getDouble(i) != 0.0;
        }
        return new SimpleBooleanTensor(boolsFromMask, shape);
    }

    private static class Nd4jDoubleFlattenedView implements FlattenedView<Double> {

        INDArray tensor;

        public Nd4jDoubleFlattenedView(INDArray tensor) {
            this.tensor = tensor;
        }

        @Override
        public long size() {
            return tensor.data().length();
        }

        @Override
        public Double get(long index) {
            return tensor.data().getDouble(index);
        }

        @Override
        public Double getOrScalar(long index) {
            if (tensor.isScalar()) {
                return get(0);
            } else {
                return get(index);
            }
        }

        @Override
        public void set(long index, Double value) {
            tensor.data().put(index, value);
        }
    }

    @Override
    public double[] asFlatDoubleArray() {
        return tensor.data().asDouble();
    }

    @Override
    public int[] asFlatIntegerArray() {
        return tensor.data().asInt();
    }

    @Override
    public Double[] asFlatArray() {
        return ArrayUtils.toObject(asFlatDoubleArray());
    }
}
