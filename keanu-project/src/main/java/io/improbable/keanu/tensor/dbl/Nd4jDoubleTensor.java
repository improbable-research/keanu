package io.improbable.keanu.tensor.dbl;

import static java.util.Arrays.copyOf;

import static io.improbable.keanu.tensor.TypedINDArrayFactory.valueArrayOf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.special.Gamma;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldGreaterThan;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldGreaterThanOrEqual;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldLessThan;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.OldLessThanOrEqual;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;

import io.improbable.keanu.tensor.INDArrayExtensions;
import io.improbable.keanu.tensor.INDArrayShim;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TypedINDArrayFactory;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.SimpleBooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.intgr.Nd4jIntegerTensor;

public class Nd4jDoubleTensor implements DoubleTensor {

    private static final DataBuffer.Type BUFFER_TYPE = DataBuffer.Type.DOUBLE;

    public static Nd4jDoubleTensor scalar(double scalarValue) {
        return new Nd4jDoubleTensor(TypedINDArrayFactory.scalar(scalarValue, BUFFER_TYPE));
    }

    public static Nd4jDoubleTensor create(double[] values, int[] shape) {
        return new Nd4jDoubleTensor(values, shape);
    }

    public static Nd4jDoubleTensor create(double value, int[] shape) {
        return new Nd4jDoubleTensor(valueArrayOf(shape, value, BUFFER_TYPE));
    }

    public static Nd4jDoubleTensor ones(int[] shape) {
        return new Nd4jDoubleTensor(TypedINDArrayFactory.ones(shape, BUFFER_TYPE));
    }

    public static Nd4jDoubleTensor eye(int n) {
        return new Nd4jDoubleTensor(TypedINDArrayFactory.eye(n, BUFFER_TYPE));
    }

    public static Nd4jDoubleTensor zeros(int[] shape) {
        return new Nd4jDoubleTensor(TypedINDArrayFactory.zeros(shape, BUFFER_TYPE));
    }

    public static Nd4jDoubleTensor linspace(double start, double end, int numberOfPoints) {
        return new Nd4jDoubleTensor(TypedINDArrayFactory.linspace(start, end, numberOfPoints, BUFFER_TYPE));
    }

    public static Nd4jDoubleTensor arange(double start, double end) {
        return new Nd4jDoubleTensor(TypedINDArrayFactory.arange(start, end, BUFFER_TYPE));
    }

    public static Nd4jDoubleTensor arange(double start, double end, double stepSize) {
        double stepCount = Math.ceil((end - start) / stepSize);
        INDArray arangeWithStep = TypedINDArrayFactory.arange(0, stepCount, BUFFER_TYPE).muli(stepSize).addi(start);
        return new Nd4jDoubleTensor(arangeWithStep);
    }

    private INDArray tensor;

    public Nd4jDoubleTensor(double[] data, int[] shape) {
        this.tensor = TypedINDArrayFactory.create(data, shape, BUFFER_TYPE);
    }

    public Nd4jDoubleTensor(INDArray tensor) {
        this.tensor = tensor;
    }

    @Override
    public int getRank() {
        return tensor.shape().length;
    }

    @Override
    public int[] getShape() {
        return tensor.shape();
    }

    @Override
    public long getLength() {
        return tensor.lengthLong();
    }

    @Override
    public boolean isShapePlaceholder() {
        return tensor == null;
    }

    public Double getValue(int... index) {
        return tensor.getDouble(index);
    }

    public DoubleTensor setValue(Double value, int... index) {
        tensor.putScalar(index, value);
        return this;
    }

    @Override
    public DoubleTensor reshape(int... newShape) {
        return new Nd4jDoubleTensor(tensor.reshape(newShape));
    }

    @Override
    public DoubleTensor permute(int... rearrange) {
        return new Nd4jDoubleTensor(tensor.permute(rearrange));
    }

    @Override
    public DoubleTensor diag() {
        return new Nd4jDoubleTensor(Nd4j.diag(tensor));
    }

    @Override
    public DoubleTensor transpose() {
        return new Nd4jDoubleTensor(tensor.transpose());
    }

    @Override
    public DoubleTensor sum(int... overDimensions) {
        return new Nd4jDoubleTensor(tensor.sum(overDimensions));
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
    public DoubleTensor matrixInverse() {
        return new Nd4jDoubleTensor(InvertMatrix.invert(tensor, false));
    }

    @Override
    public double max() {
        return tensor.maxNumber().doubleValue();
    }

    @Override
    public DoubleTensor min(DoubleTensor min) {
        return duplicate().minInPlace(min);
    }

    @Override
    public double min() {
        return tensor.minNumber().doubleValue();
    }

    @Override
    public double average() {
        return tensor.sumNumber().doubleValue() / tensor.length();
    }

    @Override
    public double standardDeviation() {
        return tensor.stdNumber().doubleValue();
    }

    @Override
    public boolean equalsWithinEpsilon(DoubleTensor o, double epsilon) {
        if (this == o) return true;

        if (o instanceof Nd4jDoubleTensor) {
            return tensor.equalsWithEps(((Nd4jDoubleTensor) o).tensor, epsilon);
        } else {
            if (this.hasSameShapeAs(o)) {
                DoubleTensor difference = o.minus(this);
                return difference.abs().lessThan(epsilon).allTrue();
            }
        }
        return false;
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
    public DoubleTensor round() {
        return duplicate().roundInPlace();
    }

    @Override
    public DoubleTensor standardize() {
        return duplicate().standardizeInPlace();
    }

    @Override
    public DoubleTensor sigmoid() {
        return duplicate().sigmoidInPlace();
    }

    @Override
    public DoubleTensor choleskyDecomposition() {
        INDArray dup = tensor.dup();
        Nd4j.getBlasWrapper().lapack().potrf(dup, false);
        return new Nd4jDoubleTensor(dup);
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
    public DoubleTensor matrixMultiply(DoubleTensor value) {
        INDArray mmulResult = tensor.mmul(unsafeGetNd4J(value));
        return new Nd4jDoubleTensor(mmulResult);
    }

    @Override
    public DoubleTensor tensorMultiply(DoubleTensor value, int[] dimsLeft, int[] dimsRight) {
        INDArray tensorMmulResult = Nd4j.tensorMmul(tensor, unsafeGetNd4J(value), new int[][]{dimsLeft, dimsRight});
        return new Nd4jDoubleTensor(tensorMmulResult);
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
    public DoubleTensor logGamma() {
        return duplicate().logGammaInPlace();
    }

    @Override
    public DoubleTensor digamma() {
        return duplicate().digammaInPlace();
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
            return this.duplicate().minusInPlace(that);
        }
    }

    @Override
    public DoubleTensor plus(DoubleTensor that) {
        if (that.isScalar()) {
            return this.plus(that.scalar());
        } else if (this.isScalar()) {
            return that.plus(this.scalar());
        } else {
            return this.duplicate().plusInPlace(that);
        }
    }

    @Override
    public DoubleTensor times(DoubleTensor that) {
        if (that.isScalar()) {
            return this.times(that.scalar());
        } else if (this.isScalar()) {
            return that.times(this.scalar());
        } else {
            return this.duplicate().timesInPlace(that);
        }
    }

    @Override
    public DoubleTensor div(DoubleTensor that) {
        if (that.isScalar()) {
            return this.div(that.scalar());
        } else if (this.isScalar()) {
            return that.reciprocal().timesInPlace(this);
        } else {
            return this.duplicate().divInPlace(that);
        }
    }

    @Override
    public DoubleTensor abs() {
        return duplicate().absInPlace();
    }

    @Override
    public DoubleTensor unaryMinus() {
        return duplicate().unaryMinusInPlace();
    }

    @Override
    public DoubleTensor setWithMask(DoubleTensor mask, Double value) {
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
    public DoubleTensor logGammaInPlace() {
        return applyInPlace(Gamma::logGamma);
    }

    @Override
    public DoubleTensor digammaInPlace() {
        return applyInPlace(Gamma::digamma);
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
        return atan2InPlace(DoubleTensor.create(y, this.tensor.shape()));
    }

    @Override
    public DoubleTensor atan2InPlace(DoubleTensor y) {
        if (y.isScalar()) {
            tensor = Transforms.atan2(tensor, valueArrayOf(this.tensor.shape(), y.scalar(), BUFFER_TYPE));
        } else {
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

    /**
     * @param that Right operand.
     * @return A new DoubleTensor instance only if <i>this</i> has a length of 1 and right operand has a length greater than 1.
     * Otherwise return <i>this</i>.
     */
    @Override
    public DoubleTensor minusInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.subi(that.scalar());
        } else if (this.isScalar()) {
            return this.minus(that);
        } else {
            INDArray result = INDArrayShim.subi(tensor, unsafeGetNd4J(that));
            if (result != tensor) {
                return new Nd4jDoubleTensor(result);
            }
        }
        return this;
    }

    /**
     * @param that Right operand.
     * @return A new DoubleTensor instance only if <i>this</i> has a length of 1 and right operand has a length greater than 1.
     * Otherwise return <i>this</i>.
     */
    @Override
    public DoubleTensor plusInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.addi(that.scalar());
        } else if (this.isScalar()) {
            return this.plus(that);
        } else {
            INDArray result = INDArrayShim.addi(tensor, unsafeGetNd4J(that));
            if (result != tensor) {
                return new Nd4jDoubleTensor(result);
            }
        }
        return this;
    }

    /**
     * @param that Right operand.
     * @return A new DoubleTensor instance only if <i>this</i> has a length of 1 and right operand has a length greater than 1.
     * Otherwise return <i>this</i>.
     */
    @Override
    public DoubleTensor timesInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.muli(that.scalar());
        } else if (this.isScalar()) {
            return this.times(that);
        } else {
            INDArray result = INDArrayShim.muli(tensor, unsafeGetNd4J(that));
            if (result != tensor) {
                return new Nd4jDoubleTensor(result);
            }
        }
        return this;
    }

    /**
     * @param that Right operand.
     * @return A new DoubleTensor instance only if <i>this</i> has a length of 1 and right operand has a length greater than 1.
     * Otherwise return <i>this</i>.
     */
    @Override
    public DoubleTensor divInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.divi(that.scalar());
        } else if (this.isScalar()) {
            return this.div(that);
        } else {
            INDArray result = INDArrayShim.divi(tensor, unsafeGetNd4J(that));
            if (result != tensor) {
                return new Nd4jDoubleTensor(result);
            }
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
        Transforms.abs(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor getGreaterThanMask(DoubleTensor greaterThanThis) {

        INDArray mask = tensor.dup();

        if (greaterThanThis.isScalar()) {
            Nd4j.getExecutioner().exec(
                new OldGreaterThan(mask,
                    valueArrayOf(mask.shape(), greaterThanThis.scalar(), BUFFER_TYPE),
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
                    valueArrayOf(mask.shape(), greaterThanOrEqualToThis.scalar(), BUFFER_TYPE),
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
                    valueArrayOf(mask.shape(), lessThanThis.scalar(), BUFFER_TYPE),
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
                    valueArrayOf(mask.shape(), lessThanOrEqualToThis.scalar(), BUFFER_TYPE),
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
    public DoubleTensor setWithMaskInPlace(DoubleTensor mask, Double value) {

        INDArray maskDup = unsafeGetNd4J(mask).dup();

        if (value == 0.0) {
            INDArray swapOnesForZeros = maskDup.rsub(1.0);
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
    public DoubleTensor standardizeInPlace() {
        tensor.subi(average()).divi(standardDeviation());
        return this;
    }

    @Override
    public DoubleTensor setAllInPlace(double value) {
        this.tensor.assign(value);
        return this;
    }

    @Override
    public DoubleTensor clampInPlace(DoubleTensor min, DoubleTensor max) {
        return minInPlace(max).maxInPlace(min);
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
    public DoubleTensor roundInPlace() {
        Transforms.round(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor sigmoidInPlace() {
        Transforms.sigmoid(tensor, false);
        return this;
    }

    @Override
    public double determinant() {
        INDArray dup = tensor.dup();
        double[][] asMatrix = dup.toDoubleMatrix();
        RealMatrix matrix = new Array2DRowRealMatrix(asMatrix);
        return new LUDecomposition(matrix).getDeterminant();
    }

    @Override
    public double product() {
        return tensor.prod().getDouble(0);
    }

    @Override
    public DoubleTensor slice(int dimension, int index) {
        INDArray dup = tensor.dup();
        INDArray slice = dup.slice(index, dimension);
        return new Nd4jDoubleTensor(slice);
    }

    /**
     * @param dimension      the dimension to slice on
     * @param splitAtIndices the indices that the dimension to slice on should be slice on
     * @return pieces of the tensor slice in the order specified by splitAtIndices. To get
     * pieces that encompasses the entire tensor, the last index in the splitAtIndices must
     * be the length of the dimension being slice on.
     * <p>
     * e.g A =
     * [
     * 1, 2, 3, 4, 5, 6
     * 7, 8, 9, 1, 2, 3
     * ]
     * <p>
     * A.slice(0, [1]) gives List([1, 2, 3, 4, 5, 6])
     * A.slice(0, [1, 2]) gives List([1, 2, 3, 4, 5, 6], [7, 8, 9, 1, 2, 3]
     * <p>
     * A.slice(1, [1, 3, 6]) gives
     * List(
     * [1, [2, 3  , [4, 5, 6,
     * 7]  8, 9]    1, 2, 3]
     * )
     */
    @Override
    public List<DoubleTensor> split(int dimension, int[] splitAtIndices) {

        int[] shape = getShape();
        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Invalid dimension to split on " + dimension);
        }

        Nd4j.getCompressor().autoDecompress(tensor);

        List<DoubleTensor> splits = new ArrayList<>();
        int previousSplitIndex = 0;
        for (int i = 0; i < splitAtIndices.length; i++) {

            INDArrayIndex[] indices = new INDArrayIndex[tensor.rank()];

            if (previousSplitIndex == splitAtIndices[i]) {
                throw new IllegalArgumentException("Invalid index to split on " + splitAtIndices[i] + " at dimension " + dimension + " for tensor of shape " + Arrays.toString(shape));
            }

            indices[dimension] = NDArrayIndex.interval(previousSplitIndex, splitAtIndices[i]);
            previousSplitIndex = splitAtIndices[i];

            for (int j = 0; j < tensor.rank(); j++) {
                if (j != dimension) {
                    indices[j] = NDArrayIndex.all();
                }
            }

            splits.add(new Nd4jDoubleTensor(tensor.get(indices)));
        }

        return splits;
    }

    // Comparisons

    @Override
    public BooleanTensor lessThan(double value) {
        return fromMask(tensor.lt(value), copyOf(getShape(), getShape().length));
    }

    @Override
    public BooleanTensor lessThanOrEqual(double value) {
        return fromMask(tensor.lte(value), copyOf(getShape(), getShape().length));
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

        return fromMask(mask, copyOf(getShape(), getShape().length));
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

        return fromMask(mask, copyOf(getShape(), getShape().length));
    }

    @Override
    public BooleanTensor greaterThan(double value) {
        return fromMask(tensor.gt(value), copyOf(getShape(), getShape().length));
    }

    @Override
    public BooleanTensor greaterThanOrEqual(double value) {
        return fromMask(tensor.gte(value), copyOf(getShape(), getShape().length));
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

        return fromMask(mask, copyOf(getShape(), getShape().length));
    }

    @Override
    public BooleanTensor greaterThanOrEqual(DoubleTensor value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.gte(value.scalar());
        } else {
            INDArray indArray = unsafeGetNd4J(value);
            mask = tensor.dup();
            Nd4j.getExecutioner().exec(new OldGreaterThanOrEqual(mask, indArray, mask, getLength()));
        }

        return fromMask(mask, copyOf(getShape(), getShape().length));
    }

    @Override
    public BooleanTensor elementwiseEquals(Tensor that) {

        if (that instanceof Nd4jDoubleTensor) {
            INDArray eq = tensor.eq(unsafeGetNd4J((Nd4jDoubleTensor) that));
            return fromMask(eq, getShape());
        } else {
            return Tensor.elementwiseEquals(this, that);
        }
    }

    static INDArray unsafeGetNd4J(DoubleTensor that) {
        if (that.isScalar()) {
            return TypedINDArrayFactory.scalar(that.scalar(), BUFFER_TYPE).reshape(that.getShape());
        }
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
            if (!Arrays.equals(that.getShape(), getShape())) return false;
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
        return duplicate();
    }

    @Override
    public IntegerTensor toInteger() {
        return new Nd4jIntegerTensor(INDArrayExtensions.castToInteger(tensor, true));
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
        return tensor.dup().data().asDouble();
    }

    @Override
    public int[] asFlatIntegerArray() {
        return tensor.dup().data().asInt();
    }

    @Override
    public Double[] asFlatArray() {
        return ArrayUtils.toObject(asFlatDoubleArray());
    }
}
