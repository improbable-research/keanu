package io.improbable.keanu.tensor.dbl;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.JVMBuffer;
import io.improbable.keanu.tensor.JVMTensorBroadcast;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.validate.TensorValidator;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShape.convertFromFlatIndexToPermutedFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.dimensionRange;
import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;
import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getPermutationForDimensionToDimensionZero;
import static io.improbable.keanu.tensor.TensorShape.getPermutedIndices;
import static io.improbable.keanu.tensor.TensorShape.getReshapeAllowingWildcard;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static io.improbable.keanu.tensor.TensorShape.getSummationResultShape;
import static io.improbable.keanu.tensor.TensorShape.incrementIndexByShape;
import static io.improbable.keanu.tensor.TensorShape.invertedPermute;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesMatch;
import static io.improbable.keanu.tensor.dbl.BroadcastableDoubleOperations.ADD;
import static io.improbable.keanu.tensor.dbl.BroadcastableDoubleOperations.DIV;
import static io.improbable.keanu.tensor.dbl.BroadcastableDoubleOperations.GTE_MASK;
import static io.improbable.keanu.tensor.dbl.BroadcastableDoubleOperations.GT_MASK;
import static io.improbable.keanu.tensor.dbl.BroadcastableDoubleOperations.LTE_MASK;
import static io.improbable.keanu.tensor.dbl.BroadcastableDoubleOperations.LT_MASK;
import static io.improbable.keanu.tensor.dbl.BroadcastableDoubleOperations.MUL;
import static io.improbable.keanu.tensor.dbl.BroadcastableDoubleOperations.RDIV;
import static io.improbable.keanu.tensor.dbl.BroadcastableDoubleOperations.RSUB;
import static io.improbable.keanu.tensor.dbl.BroadcastableDoubleOperations.SUB;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.broadcastIfNeeded;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dgetrf;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dgetri;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dpotrf;
import static java.util.Arrays.copyOf;
import static org.bytedeco.javacpp.openblas.CblasNoTrans;
import static org.bytedeco.javacpp.openblas.CblasRowMajor;
import static org.bytedeco.javacpp.openblas.cblas_dgemm;

public class JVMDoubleTensor extends DoubleTensor {

    private static final JVMBuffer.DoubleArrayWrapperFactory factory = new JVMBuffer.DoubleArrayWrapperFactory();

    private long[] shape;
    private long[] stride;
    private JVMBuffer.PrimitiveDoubleWrapper buffer;

    private JVMDoubleTensor(JVMBuffer.PrimitiveDoubleWrapper buffer, long[] shape, long[] stride) {
        this.shape = shape;
        this.stride = stride;
        this.buffer = buffer;
    }

    private JVMDoubleTensor(JVMBuffer.PrimitiveDoubleWrapper buffer, long[] shape) {
        this.shape = shape;
        this.stride = getRowFirstStride(shape);
        this.buffer = buffer;
    }

    private JVMDoubleTensor(double[] data, long[] shape, long[] stride) {
        this(factory.create(data), shape, stride);
    }

    private JVMDoubleTensor(double[] data, long[] shape) {
        this(factory.create(data), shape);
    }

    private JVMDoubleTensor(double value) {
        this.shape = new long[0];
        this.stride = new long[0];
        this.buffer = new JVMBuffer.DoubleWrapper(value);
    }

    public static JVMDoubleTensor scalar(double scalarValue) {
        return new JVMDoubleTensor(scalarValue);
    }

    public static JVMDoubleTensor create(double[] values, long... shape) {
        if (values.length != TensorShape.getLength(shape)) {
            throw new IllegalArgumentException("Shape " + Arrays.toString(shape) + " does not match buffer size " + values.length);
        }
        return new JVMDoubleTensor(values, shape);
    }

    public static JVMDoubleTensor create(double value, long... shape) {
        final int length = TensorShape.getLengthAsInt(shape);

        if (length > 1) {
            final double[] buffer = new double[length];
            if (value != 0) {
                Arrays.fill(buffer, value);
            }

            return new JVMDoubleTensor(buffer, shape);
        } else {
            return new JVMDoubleTensor(new JVMBuffer.DoubleWrapper(value), shape);
        }
    }

    public static JVMDoubleTensor ones(long... shape) {
        return create(1.0, shape);
    }

    public static JVMDoubleTensor zeros(long... shape) {
        return create(0.0, shape);
    }

    public static JVMDoubleTensor eye(long n) {

        if (n == 1) {
            return create(1.0, 1, 1);
        } else {

            double[] buffer = new double[Ints.checkedCast(n * n)];
            int nInt = Ints.checkedCast(n);
            for (int i = 0; i < n; i++) {
                buffer[i * nInt + i] = 1;
            }
            return new JVMDoubleTensor(buffer, new long[]{n, n});
        }
    }

    public static JVMDoubleTensor arange(double start, double end) {
        return arange(start, end, 1.0);
    }

    public static JVMDoubleTensor arange(double start, double end, double stepSize) {
        Preconditions.checkArgument(stepSize != 0);
        int steps = (int) Math.ceil((end - start) / stepSize);

        return linearBufferCreate(start, steps, stepSize);
    }

    public static JVMDoubleTensor linspace(double start, double end, int numberOfPoints) {
        Preconditions.checkArgument(numberOfPoints > 0);
        double stepSize = (end - start) / (numberOfPoints - 1);

        return linearBufferCreate(start, numberOfPoints, stepSize);
    }

    private static JVMDoubleTensor linearBufferCreate(double start, int numberOfPoints, double stepSize) {
        Preconditions.checkArgument(numberOfPoints > 0);
        double[] buffer = new double[numberOfPoints];

        double currentValue = start;
        for (int i = 0; i < buffer.length; i++, currentValue += stepSize) {
            buffer[i] = currentValue;
        }

        return new JVMDoubleTensor(buffer, new long[]{buffer.length});
    }

    private long[] shapeCopy() {
        return copyOf(shape, shape.length);
    }

    @Override
    public int getRank() {
        return shape.length;
    }

    @Override
    public long[] getShape() {
        return shapeCopy();
    }

    @Override
    public long[] getStride() {
        return copyOf(stride, stride.length);
    }

    @Override
    public long getLength() {
        return buffer.getLength();
    }

    @Override
    public DoubleTensor reshape(long... newShape) {
        return new JVMDoubleTensor(buffer.copy(), getReshapeAllowingWildcard(shape, buffer.getLength(), newShape));
    }

    @Override
    public DoubleTensor broadcast(long... toShape) {
        int outputLength = TensorShape.getLengthAsInt(toShape);
        long[] outputStride = TensorShape.getRowFirstStride(toShape);
        double[] outputBuffer = new double[outputLength];

        JVMTensorBroadcast.broadcast(buffer.asDoubleArray(), shape, stride, outputBuffer, outputStride);

        return new JVMDoubleTensor(outputBuffer, toShape, outputStride);
    }

    @Override
    public BooleanTensor elementwiseEquals(Tensor that) {
        if (that instanceof DoubleTensor) {
            if (isScalar()) {
                return that.elementwiseEquals(this.scalar());
            } else if (that.isScalar()) {
                return elementwiseEquals(((DoubleTensor) that).scalar());
            } else {
                DoubleTensor equalsMask = broadcastableBinaryDoubleOp(
                    (l, r) -> l.doubleValue() == r.doubleValue() ? 1.0 : 0.0, (DoubleTensor) that
                );

                return maskToBooleanTensor(equalsMask);
            }
        } else {
            return Tensor.elementwiseEquals(this, that);
        }
    }

    @Override
    public BooleanTensor elementwiseEquals(Double value) {
        boolean[] newBuffer = new boolean[buffer.getLength()];

        for (int i = 0; i < buffer.getLength(); i++) {
            newBuffer[i] = value == buffer.get(i).doubleValue();
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor permute(int... rearrange) {
        Preconditions.checkArgument(rearrange.length == shape.length);
        long[] resultShape = getPermutedIndices(shape, rearrange);
        long[] resultStride = getRowFirstStride(resultShape);
        JVMBuffer.PrimitiveDoubleWrapper newBuffer = factory.createNew(buffer.getLength());

        for (int flatIndex = 0; flatIndex < buffer.getLength(); flatIndex++) {

            int permutedFlatIndex = convertFromFlatIndexToPermutedFlatIndex(
                flatIndex,
                shape, stride,
                resultShape, resultStride,
                rearrange
            );

            newBuffer.set(buffer.get(flatIndex), permutedFlatIndex);
        }

        return new JVMDoubleTensor(newBuffer, resultShape);
    }

    @Override
    public DoubleTensor duplicate() {
        return new JVMDoubleTensor(buffer.copy(), shapeCopy(), Arrays.copyOf(stride, stride.length));
    }

    @Override
    public DoubleTensor toDouble() {
        return duplicate();
    }

    @Override
    public IntegerTensor toInteger() {
        return IntegerTensor.create(buffer.asIntegerArray(), shapeCopy());
    }

    @Override
    public DoubleTensor diag() {

        JVMBuffer.PrimitiveDoubleWrapper newBuffer;
        long[] newShape;
        if (getRank() == 1) {
            int n = buffer.getLength();
            newBuffer = factory.createNew(Ints.checkedCast((long) n * (long) n));
            ;
            for (int i = 0; i < n; i++) {
                newBuffer.set(buffer.get(i), i * n + i);
            }
            newShape = new long[]{n, n};
        } else if (getRank() == 2 && shape[0] == shape[1]) {
            int n = Ints.checkedCast(shape[0]);
            newBuffer = factory.createNew(n);
            for (int i = 0; i < n; i++) {
                newBuffer.set(buffer.get(i * n + i), i);
            }
            newShape = new long[]{n};
        } else {
            throw new IllegalArgumentException("Diag is only valid for vectors or square matrices");
        }

        return new JVMDoubleTensor(newBuffer, newShape);
    }

    @Override
    public DoubleTensor transpose() {
        if (shape.length < 2) {
            throw new IllegalArgumentException("Cannot transpose rank " + shape.length);
        }
        return permute(1, 0);
    }

    @Override
    public Double sum() {
        return buffer.sum();
    }

    @Override
    public DoubleTensor sum(int... overDimensions) {

        overDimensions = TensorShape.getAbsoluteDimensions(this.shape.length, overDimensions);

        long[] resultShape = getSummationResultShape(shape, overDimensions);
        long[] resultStride = getRowFirstStride(resultShape);
        JVMBuffer.PrimitiveDoubleWrapper newBuffer = factory.createNew(TensorShape.getLengthAsInt(resultShape));

        for (int i = 0; i < buffer.getLength(); i++) {

            long[] shapeIndices = ArrayUtils.removeAll(TensorShape.getShapeIndices(shape, stride, i), overDimensions);

            int j = Ints.checkedCast(getFlatIndex(resultShape, resultStride, shapeIndices));

            newBuffer.set(newBuffer.get(j) + buffer.get(i), j);
        }

        return new JVMDoubleTensor(newBuffer, resultShape);
    }

    @Override
    public DoubleTensor cumSumInPlace(int requestedDimension) {

        int dimension = requestedDimension >= 0 ? requestedDimension : requestedDimension + shape.length;
        TensorShapeValidation.checkDimensionExistsInShape(dimension, shape);
        long[] index = new long[shape.length];
        int[] dimensionOrder = ArrayUtils.remove(dimensionRange(0, shape.length), dimension);

        do {

            double sum = 0.0;
            for (int i = 0; i < shape[dimension]; i++) {

                index[dimension] = i;

                int j = Ints.checkedCast(getFlatIndex(shape, stride, index));
                buffer.set(buffer.get(j) + sum, j);
                sum = buffer.get(j);
            }

        } while (incrementIndexByShape(shape, index, dimensionOrder));

        return this;
    }

    @Override
    public DoubleTensor choleskyDecomposition() {

        if (shape.length != 2 || shape[0] != shape[1]) {
            throw new IllegalArgumentException("Cholesky decomposition must be performed on square matrix.");
        }

        int N = Ints.checkedCast(shape[0]);
        double[] newBuffer = buffer.copy().asDoubleArray();

        int result = dpotrf(KeanuLapack.Triangular.LOWER, N, newBuffer);

        if (result != 0) {
            throw new IllegalStateException("Cholesky decomposition failed");
        }

        zeroOutUpperTriangle(N, newBuffer);

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    private void zeroOutUpperTriangle(int N, double[] buffer) {
        if (N > 1) {
            for (int i = 0; i < N; i++) {
                for (int j = i + 1; j < N; j++) {
                    buffer[i * N + j] = 0;
                }
            }
        }
    }

    @Override
    public Double determinant() {

        final int m = Ints.checkedCast(shape[0]);
        final int n = Ints.checkedCast(shape[1]);
        final double[] newBuffer = buffer.copy().asDoubleArray();
        final int[] ipiv = new int[newBuffer.length];

        final int factorizationResult = dgetrf(m, n, newBuffer, ipiv);

        if (factorizationResult < 0) {
            throw new IllegalStateException("Matrix factorization failed");
        } else if (factorizationResult > 0) {
            return 0.0;
        }

        //credit: https://stackoverflow.com/questions/47315471/compute-determinant-from-lu-decomposition-in-lapack
        int j;
        double detp = 1.;
        for (j = 0; j < n; j++) {
            if (j + 1 != ipiv[j]) {
                detp = -detp;
            }
        }

        double detU = 1.0;
        for (int i = 0; i < m; i++) {
            detU *= newBuffer[i * m + i];
        }

        return detU * detp;
    }

    @Override
    public DoubleTensor matrixInverse() {

        final int m = Ints.checkedCast(shape[0]);
        final int n = Ints.checkedCast(shape[1]);
        final double[] newBuffer = buffer.copy().asDoubleArray();
        final int[] ipiv = new int[newBuffer.length];

        final int factorizationResult = dgetrf(m, n, newBuffer, ipiv);

        if (factorizationResult < 0) {
            throw new IllegalStateException("Matrix factorization failed");
        } else if (factorizationResult > 0) {
            throw new SingularMatrixException();
        }

        int inverseResult = dgetri(m, newBuffer, ipiv);

        if (inverseResult != 0) {
            throw new IllegalStateException("Matrix inverse failed");
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor matrixMultiply(DoubleTensor that) {

        final long[] thatShape = that.getShape();
        TensorShapeValidation.getMatrixMultiplicationResultingShape(shape, thatShape);

        //C = alpha*A*B + beta*C
        //(M,N) = (M,k)(k,N) + (M,N)
        final double[] A = buffer.asDoubleArray();
        final double[] B = getRawBufferIfJVMTensor(that).asDoubleArray();
        final double[] C = new double[Ints.checkedCast(this.shape[0] * thatShape[1])];

        final int N = Ints.checkedCast(thatShape[1]);
        final int M = Ints.checkedCast(this.shape[0]);
        final int K = Ints.checkedCast(this.shape[1]);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);

        return new JVMDoubleTensor(C, new long[]{this.shape[0], thatShape[1]});
    }

    @Override
    public DoubleTensor tensorMultiply(DoubleTensor that, int[] dimsLeft, int[] dimsRight) {
        return TensorMulByMatrixMul.tensorMmul(this, that, dimsLeft, dimsRight);
    }

    @Override
    public int argMax() {

        double max = -Double.MAX_VALUE;
        int argMax = 0;
        for (int i = 0; i < buffer.getLength(); i++) {
            final double value = buffer.get(i);
            if (value > max) {
                max = value;
                argMax = i;
            }
        }

        return argMax;
    }

    @Override
    public IntegerTensor argMax(int axis) {

        if (axis >= shape.length) {
            throw new IllegalArgumentException("Cannot take arg max of axis " + axis + " on a " + shape.length + " rank tensor.");
        }

        int[] rearrange = getPermutationForDimensionToDimensionZero(axis, shape);

        JVMDoubleTensor permuted = (JVMDoubleTensor) permute(rearrange);
        JVMBuffer.PrimitiveDoubleWrapper permutedBuffer = permuted.buffer;

        int dimLength = (int) (buffer.getLength() / shape[axis]);

        double[] maxBuffer = new double[dimLength];
        int[] maxIndex = new int[dimLength];

        for (int i = 0; i < permutedBuffer.getLength(); i++) {

            final int bufferIndex = i % dimLength;

            if (permutedBuffer.get(i) > maxBuffer[bufferIndex]) {
                maxBuffer[bufferIndex] = permutedBuffer.get(i);
                maxIndex[bufferIndex] = i / dimLength;
            }

        }

        return IntegerTensor.create(maxIndex, ArrayUtils.remove(shape, axis));
    }

    @Override
    public DoubleTensor unaryMinusInPlace() {
        buffer.apply((v) -> -v);
        return this;
    }

    @Override
    public DoubleTensor absInPlace() {
        buffer.apply(Math::abs);
        return this;
    }

    @Override
    public DoubleTensor applyInPlace(Function<Double, Double> function) {
        buffer.apply(function);
        return this;
    }

    @Override
    public DoubleTensor getGreaterThanMask(DoubleTensor greaterThanThis) {
        return broadcastableBinaryDoubleOp(GT_MASK, greaterThanThis);
    }

    @Override
    public DoubleTensor getGreaterThanOrEqualToMask(DoubleTensor greaterThanThis) {
        return broadcastableBinaryDoubleOp(GTE_MASK, greaterThanThis);
    }

    @Override
    public DoubleTensor getLessThanMask(DoubleTensor lessThanThis) {
        return broadcastableBinaryDoubleOp(LT_MASK, lessThanThis);
    }

    @Override
    public DoubleTensor getLessThanOrEqualToMask(DoubleTensor lessThanThis) {
        return broadcastableBinaryDoubleOp(LTE_MASK, lessThanThis);
    }

    @Override
    public DoubleTensor setWithMaskInPlace(DoubleTensor mask, Double value) {
        checkMaskLengthMatches(mask);

        JVMBuffer.PrimitiveDoubleWrapper maskBuffer = getRawBufferIfJVMTensor(mask);

        for (int i = 0; i < buffer.getLength(); i++) {
            if (maskBuffer.get(i) == 1.0) {
                buffer.set(value, i);
            }
        }

        return this;
    }

    private void checkMaskLengthMatches(DoubleTensor mask) {
        if (getLength() != mask.getLength()) {
            throw new IllegalArgumentException(
                "The lengths of the tensor and mask must match, but got tensor length: " + getLength()
                    + ", mask length: " + mask.getLength()
            );
        }
    }

    @Override
    public DoubleTensor setWithMask(DoubleTensor mask, Double value) {
        checkShapesMatch(shape, mask.getShape());

        JVMBuffer.PrimitiveDoubleWrapper newBuffer = factory.createNew(buffer.getLength());
        JVMBuffer.PrimitiveDoubleWrapper maskBuffer = getRawBufferIfJVMTensor(mask);

        for (int i = 0; i < buffer.getLength(); i++) {
            newBuffer.set(maskBuffer.get(i) == 1.0 ? value : buffer.get(i), i);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public BooleanTensor lessThan(DoubleTensor that) {
        return maskToBooleanTensor(getLessThanMask(that));
    }

    @Override
    public BooleanTensor lessThanOrEqual(DoubleTensor that) {
        return maskToBooleanTensor(getLessThanOrEqualToMask(that));
    }

    @Override
    public BooleanTensor greaterThan(DoubleTensor that) {
        return maskToBooleanTensor(getGreaterThanMask(that));
    }

    @Override
    public BooleanTensor greaterThanOrEqual(DoubleTensor that) {
        return maskToBooleanTensor(getGreaterThanOrEqualToMask(that));
    }

    private BooleanTensor maskToBooleanTensor(DoubleTensor mask) {
        JVMBuffer.PrimitiveDoubleWrapper maskBuffer = getRawBufferIfJVMTensor(mask);
        boolean[] boolBuffer = new boolean[maskBuffer.getLength()];

        for (int i = 0; i < maskBuffer.getLength(); i++) {
            boolBuffer[i] = maskBuffer.get(i) == 1.0;
        }

        return BooleanTensor.create(boolBuffer, mask.getShape());
    }

    @Override
    public BooleanTensor lessThan(Double value) {
        boolean[] newBuffer = new boolean[buffer.getLength()];

        for (int i = 0; i < buffer.getLength(); i++) {
            newBuffer[i] = buffer.get(i) < value;
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public BooleanTensor lessThanOrEqual(Double value) {
        boolean[] newBuffer = new boolean[buffer.getLength()];

        for (int i = 0; i < buffer.getLength(); i++) {
            newBuffer[i] = buffer.get(i) <= value;
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public BooleanTensor greaterThan(Double value) {
        boolean[] newBuffer = new boolean[buffer.getLength()];

        for (int i = 0; i < buffer.getLength(); i++) {
            newBuffer[i] = buffer.get(i) > value;
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public BooleanTensor greaterThanOrEqual(Double value) {
        boolean[] newBuffer = new boolean[buffer.getLength()];

        for (int i = 0; i < buffer.getLength(); i++) {
            newBuffer[i] = buffer.get(i) >= value;
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor powInPlace(DoubleTensor exponent) {
        return broadcastableBinaryDoubleOpInPlace(FastMath::pow, exponent);
    }

    @Override
    public DoubleTensor powInPlace(Double exponent) {
        buffer.applyRight(FastMath::pow, exponent);
        return this;
    }

    @Override
    public DoubleTensor atan2InPlace(Double y) {
        buffer.applyLeft(FastMath::atan2, y);
        return this;
    }

    @Override
    public DoubleTensor atan2InPlace(DoubleTensor y) {
        return broadcastableBinaryDoubleOpInPlace((left, right) -> FastMath.atan2(right, left), y);
    }

    @Override
    public Double average() {
        return sum() / buffer.getLength();
    }

    @Override
    public Double standardDeviation() {

        SummaryStatistics stats = new SummaryStatistics();
        for (int i = 0; i < buffer.getLength(); i++) {
            stats.addValue(buffer.get(i));
        }

        return stats.getStandardDeviation();
    }

    @Override
    public boolean equalsWithinEpsilon(DoubleTensor other, Double epsilon) {
        if (!Arrays.equals(shape, other.getShape())) {
            return false;
        }

        JVMBuffer.PrimitiveDoubleWrapper otherBuffer = getRawBufferIfJVMTensor(other);

        for (int i = 0; i < buffer.getLength(); i++) {
            if (Math.abs(buffer.get(i) - otherBuffer.get(i)) > epsilon) {
                return false;
            }
        }

        return true;
    }

    private static final Sigmoid sigmoid = new Sigmoid();

    @Override
    public DoubleTensor sigmoidInPlace() {
        buffer.apply(sigmoid::value);
        return this;
    }

    @Override
    public Double product() {
        double result = 1.0;
        for (int i = 0; i < buffer.getLength(); i++) {
            result *= buffer.get(i);
        }
        return result;
    }

    @Override
    public DoubleTensor slice(int dimension, long index) {
        Preconditions.checkArgument(dimension < shape.length && index < shape[dimension]);
        long[] resultShape = ArrayUtils.remove(shape, dimension);
        long[] resultStride = getRowFirstStride(resultShape);
        double[] newBuffer = new double[Ints.checkedCast(TensorShape.getLength(resultShape))];

        for (int i = 0; i < newBuffer.length; i++) {

            long[] shapeIndices = ArrayUtils.insert(dimension, TensorShape.getShapeIndices(resultShape, resultStride, i), index);

            int j = Ints.checkedCast(getFlatIndex(shape, stride, shapeIndices));

            newBuffer[i] = buffer.get(j);
        }

        return new JVMDoubleTensor(newBuffer, resultShape);
    }

    @Override
    public DoubleTensor take(long... index) {
        return JVMDoubleTensor.scalar(getValue(index));
    }

    public static DoubleTensor concat(int dimension, DoubleTensor... toConcat) {

        long[] concatShape = TensorShape.getConcatResultShape(dimension, toConcat);

        boolean shouldRearrange = dimension != 0;

        if (shouldRearrange) {

            int[] rearrange = getPermutationForDimensionToDimensionZero(dimension, concatShape);

            NumberTensor[] toConcatOnDimensionZero = new NumberTensor[toConcat.length];

            for (int i = 0; i < toConcatOnDimensionZero.length; i++) {
                toConcatOnDimensionZero[i] = toConcat[i].permute(rearrange);
            }

            long[] permutedConcatShape = getPermutedIndices(concatShape, rearrange);
            DoubleTensor concatOnDimZero = concatOnDimensionZero(permutedConcatShape, toConcatOnDimensionZero);

            return concatOnDimZero.permute(invertedPermute(rearrange));
        } else {

            return concatOnDimensionZero(concatShape, toConcat);
        }
    }

    private static JVMDoubleTensor concatOnDimensionZero(long[] concatShape, NumberTensor... toConcat) {

        double[] concatBuffer = new double[TensorShape.getLengthAsInt(concatShape)];
        int bufferPosition = 0;

        for (int i = 0; i < toConcat.length; i++) {

            JVMBuffer.PrimitiveDoubleWrapper cBuffer = getRawBufferIfJVMTensor(toConcat[i]);
            System.arraycopy(cBuffer.asDoubleArray(), 0, concatBuffer, bufferPosition, cBuffer.getLength());
            bufferPosition += cBuffer.getLength();
        }

        return new JVMDoubleTensor(concatBuffer, concatShape);
    }

    @Override
    public double[] asFlatDoubleArray() {
        return buffer.copy().asDoubleArray();
    }

    private static JVMBuffer.PrimitiveDoubleWrapper getRawBufferIfJVMTensor(NumberTensor tensor) {
        if (tensor instanceof JVMDoubleTensor) {
            return ((JVMDoubleTensor) tensor).buffer;
        } else {
            return new JVMBuffer.DoubleArrayWrapper(tensor.asFlatDoubleArray());
        }
    }

    @Override
    public int[] asFlatIntegerArray() {
        return buffer.asIntegerArray();
    }

    @Override
    public Double[] asFlatArray() {
        Double[] boxedBuffer = new Double[buffer.getLength()];
        for (int i = 0; i < buffer.getLength(); i++) {
            boxedBuffer[i] = buffer.get(i);
        }
        return boxedBuffer;
    }

    @Override
    public List<DoubleTensor> split(int dimension, long... splitAtIndices) {
        dimension = getAbsoluteDimension(dimension, getRank());

        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Invalid dimension to split on " + dimension);
        }

        int[] moveDimToZero = TensorShape.slideDimension(dimension, 0, shape.length);
        int[] moveZeroToDim = TensorShape.slideDimension(0, dimension, shape.length);

        JVMDoubleTensor permutedTensor = (JVMDoubleTensor) this.permute(moveDimToZero);

        JVMBuffer.PrimitiveDoubleWrapper rawBuffer = permutedTensor.buffer;

        List<DoubleTensor> splitTensor = new ArrayList<>();

        long previousSplitAtIndex = 0;
        int rawBufferPosition = 0;
        for (long splitAtIndex : splitAtIndices) {

            long[] subTensorShape = shapeCopy();
            long subTensorLengthInDimension = splitAtIndex - previousSplitAtIndex;

            if (subTensorLengthInDimension > shape[dimension] || subTensorLengthInDimension <= 0) {
                throw new IllegalArgumentException("Invalid index to split on " + splitAtIndex + " at " + dimension + " for tensor of shape " + Arrays.toString(shape));
            }

            subTensorShape[dimension] = subTensorLengthInDimension;
            int subTensorLength = Ints.checkedCast(TensorShape.getLength(subTensorShape));

            double[] buffer = new double[subTensorLength];
            System.arraycopy(rawBuffer.asDoubleArray(), rawBufferPosition, buffer, 0, buffer.length);

            long[] subTensorPermutedShape = getPermutedIndices(subTensorShape, moveDimToZero);
            DoubleTensor subTensor = DoubleTensor.create(buffer, subTensorPermutedShape).permute(moveZeroToDim);
            splitTensor.add(subTensor);

            previousSplitAtIndex = splitAtIndex;
            rawBufferPosition += buffer.length;
        }

        return splitTensor;
    }

    @Override
    public DoubleTensor reciprocalInPlace() {
        buffer.apply((v) -> 1.0 / v);
        return this;
    }


    @Override
    public DoubleTensor sqrtInPlace() {
        buffer.apply(FastMath::sqrt);
        return this;
    }

    @Override
    public DoubleTensor logInPlace() {
        buffer.apply(FastMath::log);
        return this;
    }

    @Override
    public DoubleTensor safeLogTimesInPlace(DoubleTensor y) {
        TensorValidator.NAN_CATCHER.validate(this);
        TensorValidator.NAN_CATCHER.validate(y);
        DoubleTensor result = this.logInPlace().timesInPlace(y);
        return TensorValidator.NAN_FIXER.validate(result);
    }

    @Override
    public DoubleTensor logGammaInPlace() {
        buffer.apply(Gamma::logGamma);
        return this;
    }

    @Override
    public DoubleTensor digammaInPlace() {
        buffer.apply(Gamma::digamma);
        return this;
    }

    @Override
    public DoubleTensor sinInPlace() {
        buffer.apply(FastMath::sin);
        return this;
    }

    @Override
    public DoubleTensor cosInPlace() {
        buffer.apply(FastMath::cos);
        return this;
    }

    @Override
    public DoubleTensor tanInPlace() {
        buffer.apply(FastMath::tan);
        return this;
    }

    @Override
    public DoubleTensor atanInPlace() {
        buffer.apply(FastMath::atan);
        return this;
    }

    @Override
    public DoubleTensor asinInPlace() {
        buffer.apply(FastMath::asin);
        return this;
    }

    @Override
    public DoubleTensor acosInPlace() {
        buffer.apply(FastMath::acos);
        return this;
    }

    @Override
    public DoubleTensor expInPlace() {
        buffer.apply(FastMath::exp);
        return this;
    }

    @Override
    public Double min() {
        double result = Double.MAX_VALUE;
        for (int i = 0; i < buffer.getLength(); i++) {
            result = Math.min(result, buffer.get(i));
        }
        return result;
    }

    @Override
    public DoubleTensor minInPlace(DoubleTensor that) {
        return broadcastableBinaryDoubleOpInPlace(Math::min, that);
    }

    @Override
    public Double max() {
        double result = -Double.MAX_VALUE;
        for (int i = 0; i < buffer.getLength(); i++) {
            result = Math.max(result, buffer.get(i));
        }
        return result;
    }

    @Override
    public DoubleTensor maxInPlace(DoubleTensor that) {
        return broadcastableBinaryDoubleOpInPlace(Math::max, that);
    }

    @Override
    public DoubleTensor clampInPlace(DoubleTensor min, DoubleTensor max) {
        maxInPlace(min);
        minInPlace(max);
        return this;
    }

    @Override
    public DoubleTensor ceilInPlace() {
        buffer.apply(FastMath::ceil);
        return this;
    }

    @Override
    public DoubleTensor floorInPlace() {
        buffer.apply(FastMath::floor);
        return this;
    }

    @Override
    public DoubleTensor roundInPlace() {
        for (int i = 0; i < buffer.getLength(); i++) {
            if (buffer.get(i) >= 0.0) {
                buffer.set(FastMath.floor(buffer.get(i) + 0.5), i);
            } else {
                buffer.set(FastMath.ceil(buffer.get(i) - 0.5), i);
            }
        }

        return this;
    }

    @Override
    public DoubleTensor standardizeInPlace() {
        return this.minusInPlace(average()).divInPlace(standardDeviation());
    }

    @Override
    public DoubleTensor replaceNaNInPlace(Double value) {
        for (int i = 0; i < buffer.getLength(); i++) {
            buffer.set(Double.isNaN(buffer.get(i)) ? value : buffer.get(i), i);
        }
        return this;
    }

    @Override
    public DoubleTensor setAllInPlace(Double value) {
        for (int i = 0; i < buffer.getLength(); i++) {
            buffer.set(value, i);
        }
        return this;
    }

    @Override
    public BooleanTensor notNaN() {
        boolean[] newBuffer = new boolean[buffer.getLength()];

        for (int i = 0; i < buffer.getLength(); i++) {
            newBuffer[i] = !Double.isNaN(buffer.get(i));
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor minusInPlace(Double value) {
        buffer.applyRight(SUB, value);
        return this;
    }

    @Override
    public DoubleTensor minusInPlace(DoubleTensor that) {
        return broadcastableBinaryDoubleOpInPlace(SUB, that);
    }

    @Override
    public DoubleTensor reverseMinusInPlace(DoubleTensor value) {
        return broadcastableBinaryDoubleOpInPlace(RSUB, value);
    }

    @Override
    public DoubleTensor reverseMinusInPlace(Double value) {
        buffer.applyRight(RSUB, value);
        return this;
    }

    @Override
    public DoubleTensor plusInPlace(Double value) {
        buffer.applyRight(ADD, value);
        return this;
    }

    @Override
    public DoubleTensor plusInPlace(DoubleTensor that) {
        return broadcastableBinaryDoubleOpInPlace(ADD, that);
    }

    @Override
    public DoubleTensor timesInPlace(Double value) {
        buffer.applyRight(MUL, value);
        return this;
    }

    @Override
    public DoubleTensor timesInPlace(DoubleTensor that) {
        return broadcastableBinaryDoubleOpInPlace(MUL, that);
    }

    @Override
    public DoubleTensor divInPlace(Double value) {
        buffer.applyRight(DIV, value);
        return this;
    }

    @Override
    public DoubleTensor divInPlace(DoubleTensor that) {
        return broadcastableBinaryDoubleOpInPlace(DIV, that);
    }

    @Override
    public DoubleTensor reverseDivInPlace(Double value) {
        buffer.applyRight(RDIV, value);
        return this;
    }

    @Override
    public DoubleTensor reverseDivInPlace(DoubleTensor that) {
        return broadcastableBinaryDoubleOpInPlace(RDIV, that);
    }

    private DoubleTensor broadcastableBinaryDoubleOp(BiFunction<Double, Double, Double> op, DoubleTensor that) {
        return binaryDoubleOpWithAutoBroadcast(that, op, false);
    }

    private DoubleTensor broadcastableBinaryDoubleOpInPlace(BiFunction<Double, Double, Double> op, DoubleTensor that) {
        return binaryDoubleOpWithAutoBroadcast(that, op, true);
    }

    private JVMDoubleTensor binaryDoubleOpWithAutoBroadcast(DoubleTensor right,
                                                            BiFunction<Double, Double, Double> op,
                                                            boolean inPlace) {
        final JVMBuffer.PrimitiveDoubleWrapper rightBuffer = getRawBufferIfJVMTensor(right);
        final long[] rightShape = right.getShape();

        final JVMDoubleTensorBroadcast.ResultWrapper result = broadcastIfNeeded(
            factory,
            buffer, shape, stride, buffer.getLength(),
            rightBuffer, rightShape, right.getStride(), rightBuffer.getLength(),
            op, inPlace
        );

        if (inPlace) {
            this.buffer = result.outputBuffer;
            this.shape = result.outputShape;
            this.stride = result.outputStride;

            return this;
        } else {
            return new JVMDoubleTensor(result.outputBuffer, result.outputShape, result.outputStride);
        }
    }

    @Override
    public FlattenedView<Double> getFlattenedView() {
        if (buffer.getLength() == 1) {
            return new ScalarJVMFlattenedView();
        } else {
            return new TensorJVMDoubleFlattenedView();
        }
    }

    private class JVMDoubleFlattenedView {
        public long size() {
            return buffer.getLength();
        }

        public Double get(long index) {
            return buffer.get(Ints.checkedCast(index));
        }

        public void set(long index, Double value) {
            buffer.set(value, Ints.checkedCast(index));
        }

    }

    private class TensorJVMDoubleFlattenedView extends JVMDoubleFlattenedView implements FlattenedView<Double> {
        @Override
        public Double getOrScalar(long index) {
            return get(index);
        }
    }

    private class ScalarJVMFlattenedView extends JVMDoubleFlattenedView implements FlattenedView<Double> {
        @Override
        public Double getOrScalar(long index) {
            return buffer.get(0);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        JVMDoubleTensor that = (JVMDoubleTensor) o;
        return Arrays.equals(shape, that.shape) && buffer.equals(that.buffer);
    }

    @Override
    public String toString() {
        return "{" +
            "shape=" + Arrays.toString(shape) +
            ", buffer=" + Arrays.toString(buffer.asDoubleArray()) +
            '}';
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(shape);
        result = 31 * result + buffer.hashCode();
        return result;
    }

}
