package io.improbable.keanu.tensor.dbl;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.validate.TensorValidator;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.shape.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;
import static io.improbable.keanu.tensor.TensorShape.invertedPermute;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkShapesMatch;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.ADD;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.DIV;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.GTE_MASK;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.GT_MASK;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.LTE_MASK;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.LT_MASK;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.MUL;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.SUB;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.broadcastFromLeft;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.broadcastFromRight;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.scalarLeft;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.scalarRight;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dgetrf;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dgetri;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dpotrf;
import static java.util.Arrays.copyOf;
import static org.bytedeco.javacpp.openblas.CblasNoTrans;
import static org.bytedeco.javacpp.openblas.CblasRowMajor;
import static org.bytedeco.javacpp.openblas.cblas_dgemm;

public class JVMDoubleTensor extends DoubleTensor {

    private long[] shape;
    private long[] stride;
    private double[] buffer;

    private JVMDoubleTensor(double value) {
        this.shape = new long[0];
        this.stride = new long[0];
        this.buffer = new double[]{value};
    }

    private JVMDoubleTensor(double[] data, long[] shape) {

        if (data.length != TensorShape.getLength(shape)) {
            throw new IllegalArgumentException("Shape " + Arrays.toString(shape) + " does not match buffer size " + data.length);
        }

        this.shape = shape;
        this.stride = TensorShape.getRowFirstStride(shape);
        this.buffer = data;
    }

    public static JVMDoubleTensor scalar(double scalarValue) {
        return new JVMDoubleTensor(scalarValue);
    }

    public static JVMDoubleTensor create(double[] values, long... shape) {
        return new JVMDoubleTensor(values, shape);
    }

    public static JVMDoubleTensor create(double value, long... shape) {
        long length = TensorShape.getLength(shape);
        double[] buffer = new double[Ints.checkedCast(length)];

        if (value != 0) {
            Arrays.fill(buffer, value);
        }

        return new JVMDoubleTensor(buffer, shape);
    }

    public static JVMDoubleTensor ones(long... shape) {
        return create(1.0, shape);
    }

    public static JVMDoubleTensor zeros(long... shape) {
        return create(0.0, shape);
    }

    public static JVMDoubleTensor eye(long n) {

        double[] buffer = new double[Ints.checkedCast(n * n)];
        int nInt = Ints.checkedCast(n);
        for (int i = 0; i < n; i++) {
            buffer[i * nInt + i] = 1;
        }
        return new JVMDoubleTensor(buffer, new long[]{n, n});
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

    private double[] newBufferOfSameLength() {
        return new double[buffer.length];
    }

    private double[] bufferCopy() {
        return copyOf(buffer, buffer.length);
    }

    private int[] bufferAsInteger() {
        int[] intBuffer = new int[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            intBuffer[i] = (int) buffer[i];
        }
        return intBuffer;
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
        return buffer.length;
    }

    @Override
    public Double getValue(long... index) {
        long flatIndex = getFlatIndex(index);
        return buffer[Ints.checkedCast(flatIndex)];
    }

    @Override
    public DoubleTensor setValue(Double value, long... index) {
        long flatIndex = getFlatIndex(index);
        buffer[Ints.checkedCast(flatIndex)] = value;
        return this;
    }

    private long getFlatIndex(long... index) {
        long flatIndex;
        if (index.length == 1) {
            flatIndex = index[0];
        } else {
            if (index.length != shape.length) {
                throw new IllegalArgumentException("Cannot get index " + Arrays.toString(index) + " for shape " + Arrays.toString(shape));
            }
            flatIndex = TensorShape.getFlatIndex(shape, stride, index);
        }
        return flatIndex;
    }

    @Override
    public Double scalar() {
        return buffer[0];
    }

    @Override
    public DoubleTensor reshape(long... newShape) {

        long newLength = 1;
        int negativeDimension = -1;
        long newShapeCopy[] = new long[newShape.length];
        System.arraycopy(newShape, 0, newShapeCopy, 0, newShape.length);

        for (int i = 0; i < newShapeCopy.length; i++) {

            long dimILength = newShapeCopy[i];
            if (dimILength > 0) {
                newLength *= dimILength;
            } else if (dimILength < 0) {
                if (negativeDimension >= 0) {
                    throw new IllegalArgumentException("Cannot reshape " + Arrays.toString(shape) + " to " + Arrays.toString(newShapeCopy));
                }
                negativeDimension = i;
            }
        }

        if (newLength != buffer.length || negativeDimension >= 0) {
            if (negativeDimension < 0) {
                throw new IllegalArgumentException("Cannot reshape " + Arrays.toString(shape) + " to " + Arrays.toString(newShapeCopy));
            } else {
                newShapeCopy[negativeDimension] = buffer.length / newLength;
            }
        }

        return new JVMDoubleTensor(copyOf(buffer, buffer.length), newShapeCopy);
    }

    @Override
    public BooleanTensor elementwiseEquals(Tensor that) {
        if (that instanceof DoubleTensor) {
            if (isScalar()) {
                return that.elementwiseEquals(this.scalar());
            } else if (that.isScalar()) {
                return elementwiseEquals(((DoubleTensor) that).scalar());
            } else {
                DoubleTensor equalsMask = broadcastableBinaryDoubleOp((l, r) -> l.equals(r) ? 1.0 : 0.0, (DoubleTensor) that);

                return maskToBooleanTensor(equalsMask);
            }
        } else {
            return Tensor.elementwiseEquals(this, that);
        }
    }

    @Override
    public BooleanTensor elementwiseEquals(Double value) {
        boolean[] newBuffer = new boolean[buffer.length];

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = value == buffer[i];
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor permute(int... rearrange) {
        Preconditions.checkArgument(rearrange.length == shape.length);
        long[] resultShape = TensorShape.getPermutedResultShapeShape(shape, rearrange);
        long[] resultStride = TensorShape.getRowFirstStride(resultShape);
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            long[] shapeIndices = TensorShape.getShapeIndices(shape, stride, i);
            long[] permutedIndex = new long[shapeIndices.length];

            for (int p = 0; p < permutedIndex.length; p++) {
                permutedIndex[p] = shapeIndices[rearrange[p]];
            }

            int j = Ints.checkedCast(TensorShape.getFlatIndex(resultShape, resultStride, permutedIndex));

            newBuffer[j] = buffer[i];
        }

        return new JVMDoubleTensor(newBuffer, resultShape);
    }

    @Override
    public DoubleTensor duplicate() {
        return new JVMDoubleTensor(copyOf(buffer, buffer.length), shapeCopy());
    }

    @Override
    public DoubleTensor toDouble() {
        return duplicate();
    }

    @Override
    public IntegerTensor toInteger() {
        return IntegerTensor.create(bufferAsInteger(), shapeCopy());
    }

    @Override
    public DoubleTensor diag() {
        int n = buffer.length;
        double[] newBuffer = new double[Ints.checkedCast(n * n)];
        int nInt = Ints.checkedCast(n);
        for (int i = 0; i < n; i++) {
            newBuffer[i * nInt + i] = buffer[i];
        }

        return new JVMDoubleTensor(newBuffer, new long[]{n, n});
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
        double result = 0;
        for (int i = 0; i < buffer.length; i++) {
            result += buffer[i];
        }
        return result;
    }

    @Override
    public DoubleTensor sum(int... overDimensions) {

        overDimensions = TensorShape.getAbsoluteDimensions(this.shape.length, overDimensions);

        long[] resultShape = TensorShape.getSummationResultShape(shape, overDimensions);
        long[] resultStride = TensorShape.getRowFirstStride(resultShape);
        double[] newBuffer = new double[Ints.checkedCast(TensorShape.getLength(resultShape))];

        for (int i = 0; i < buffer.length; i++) {

            long[] shapeIndices = ArrayUtils.removeAll(TensorShape.getShapeIndices(shape, stride, i), overDimensions);

            int j = Ints.checkedCast(TensorShape.getFlatIndex(resultShape, resultStride, shapeIndices));

            newBuffer[j] += buffer[i];
        }

        return new JVMDoubleTensor(newBuffer, resultShape);
    }

    @Override
    public DoubleTensor reciprocal() {
        double[] result = new double[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            result[i] = 1.0 / buffer[i];
        }
        return new JVMDoubleTensor(result, shapeCopy());
    }

    @Override
    public DoubleTensor choleskyDecomposition() {

        if (shape.length != 2 || shape[0] != shape[1]) {
            throw new IllegalArgumentException("Cholesky decomposition must be performed on square matrix.");
        }

        int N = Ints.checkedCast(shape[0]);
        double[] newBuffer = bufferCopy();

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
    public double determinant() {

        final int m = Ints.checkedCast(shape[0]);
        final int n = Ints.checkedCast(shape[1]);
        final double[] newBuffer = bufferCopy();
        final int[] ipiv = new int[newBuffer.length];

        final int factorizationResult = dgetrf(m, n, newBuffer, ipiv);

        if (factorizationResult < 0) {
            throw new IllegalStateException("Matrix factorization failed");
        } else if (factorizationResult > 0) {
            return 0;
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
        final double[] newBuffer = bufferCopy();
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
        if (this.shape.length != 2 || thatShape.length != 2 || shape[1] != thatShape[0]) {
            throw new IllegalArgumentException("Cannot matrix multiply shape " + Arrays.toString(shape) + " shape " + Arrays.toString(thatShape));
        }

        //C = alpha*A*B + beta*C
        //(M,N) = (M,k)(k,N) + (M,N)
        final double[] A = buffer;
        final double[] B = getRawBufferIfJVMTensor(that);
        final double[] C = new double[Ints.checkedCast(this.shape[0] * thatShape[1])];

        final int N = (int) thatShape[1];
        final int M = (int) this.shape[0];
        final int K = (int) this.shape[1];

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);

        return new JVMDoubleTensor(C, new long[]{this.shape[0], thatShape[1]});
    }

    @Override
    public DoubleTensor tensorMultiply(DoubleTensor that, int[] dimsLeft, int[] dimsRight) {
        return TensorMulByMatrixMul.tensorMmul(this, that, new int[][]{dimsLeft, dimsRight});
    }

    @Override
    public DoubleTensor abs() {
        double[] result = new double[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            result[i] = Math.abs(buffer[i]);
        }
        return new JVMDoubleTensor(result, shapeCopy());
    }

    @Override
    public int argMax() {

        double max = -Double.MAX_VALUE;
        int argMax = 0;
        for (int i = 0; i < buffer.length; i++) {
            if (buffer[i] > max) {
                max = buffer[i];
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

        int[] rearrange = shiftDimensionToDimensionZero(axis, shape);

        JVMDoubleTensor permuted = (JVMDoubleTensor) permute(rearrange);
        double[] permutedBuffer = permuted.buffer;

        int dimLength = Ints.checkedCast(buffer.length / shape[axis]);

        double[] maxBuffer = new double[dimLength];
        int[] maxIndex = new int[dimLength];

        for (int i = 0; i < permutedBuffer.length; i++) {

            final int bufferIndex = i % dimLength;

            if (permutedBuffer[i] > maxBuffer[bufferIndex]) {
                maxBuffer[bufferIndex] = permutedBuffer[i];
                maxIndex[bufferIndex] = i / dimLength;
            }

        }

        return IntegerTensor.create(maxIndex, ArrayUtils.remove(shape, axis));
    }

    @Override
    public DoubleTensor apply(Function<Double, Double> function) {
        double[] result = new double[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            result[i] = function.apply(buffer[i]);
        }
        return new JVMDoubleTensor(result, shapeCopy());
    }

    @Override
    public DoubleTensor unaryMinus() {

        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = -buffer[i];
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor unaryMinusInPlace() {

        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = -buffer[i];
        }

        return this;
    }

    @Override
    public DoubleTensor absInPlace() {

        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = Math.abs(buffer[i]);
        }

        return this;
    }

    @Override
    public DoubleTensor applyInPlace(Function<Double, Double> function) {

        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = function.apply(buffer[i]);
        }

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
        checkShapesMatch(shape, mask.getShape());

        double[] maskBuffer = getRawBufferIfJVMTensor(mask);

        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = maskBuffer[i] == 1.0 ? value : buffer[i];
        }

        return this;
    }

    @Override
    public DoubleTensor setWithMask(DoubleTensor mask, Double value) {
        checkShapesMatch(shape, mask.getShape());

        double[] newBuffer = new double[buffer.length];
        double[] maskBuffer = getRawBufferIfJVMTensor(mask);

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = maskBuffer[i] == 1.0 ? value : buffer[i];
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
        double[] maskBuffer = getRawBufferIfJVMTensor(mask);
        boolean[] boolBuffer = new boolean[maskBuffer.length];

        for (int i = 0; i < maskBuffer.length; i++) {
            boolBuffer[i] = maskBuffer[i] == 1.0;
        }

        return BooleanTensor.create(boolBuffer, mask.getShape());
    }

    @Override
    public BooleanTensor lessThan(double value) {
        boolean[] newBuffer = new boolean[buffer.length];

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = buffer[i] < value;
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public BooleanTensor lessThanOrEqual(double value) {
        boolean[] newBuffer = new boolean[buffer.length];

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = buffer[i] <= value;
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public BooleanTensor greaterThan(double value) {
        boolean[] newBuffer = new boolean[buffer.length];

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = buffer[i] > value;
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public BooleanTensor greaterThanOrEqual(double value) {
        boolean[] newBuffer = new boolean[buffer.length];

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = buffer[i] >= value;
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor powInPlace(DoubleTensor exponent) {
        return broadcastableBinaryDoubleOpInPlace(FastMath::pow, exponent);
    }

    @Override
    public DoubleTensor pow(DoubleTensor exponent) {
        return broadcastableBinaryDoubleOp(FastMath::pow, exponent);
    }

    @Override
    public DoubleTensor powInPlace(double exponent) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.pow(buffer[i], exponent);
        }
        return this;
    }

    @Override
    public DoubleTensor pow(double exponent) {

        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.pow(buffer[i], exponent);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor sqrt() {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.sqrt(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor log() {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.log(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor safeLogTimes(DoubleTensor y) {
        return duplicate().safeLogTimesInPlace(y);
    }

    @Override
    public DoubleTensor logGamma() {

        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = Gamma.logGamma(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor digamma() {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = Gamma.digamma(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor sin() {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.sin(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor cos() {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.cos(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor tan() {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.tan(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor atan() {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.atan(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor atan2(double y) {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.atan2(y, buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor atan2InPlace(double y) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.atan2(y, buffer[i]);
        }
        return this;
    }

    @Override
    public DoubleTensor atan2InPlace(DoubleTensor y) {
        return broadcastableBinaryDoubleOpInPlace((left, right) -> FastMath.atan2(right, left), y);
    }

    @Override
    public DoubleTensor atan2(DoubleTensor y) {
        return broadcastableBinaryDoubleOp((left, right) -> FastMath.atan2(right, left), y);
    }

    @Override
    public DoubleTensor asin() {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.asin(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor acos() {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.acos(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor exp() {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.exp(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public double average() {
        return sum() / buffer.length;
    }

    @Override
    public double standardDeviation() {

        SummaryStatistics stats = new SummaryStatistics();
        for (int i = 0; i < buffer.length; i++) {
            stats.addValue(buffer[i]);
        }

        return stats.getStandardDeviation();
    }

    @Override
    public boolean equalsWithinEpsilon(DoubleTensor other, double epsilon) {
        if (!Arrays.equals(shape, other.getShape())) {
            return false;
        }

        double[] otherBuffer = getRawBufferIfJVMTensor(other);

        for (int i = 0; i < buffer.length; i++) {
            if (Math.abs(buffer[i] - otherBuffer[i]) > epsilon) {
                return false;
            }
        }

        return true;
    }

    @Override
    public DoubleTensor standardize() {
        throw new NotImplementedException("");
    }

    @Override
    public DoubleTensor replaceNaN(double value) {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = Double.isNaN(buffer[i]) ? value : buffer[i];
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor clamp(DoubleTensor min, DoubleTensor max) {
        return duplicate().clampInPlace(min, max);
    }

    @Override
    public DoubleTensor ceil() {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.ceil(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor floor() {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.floor(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    /**
     * Round half up as used in ND4j
     */
    public DoubleTensor round() {
        double[] newBuffer = newBufferOfSameLength();

        for (int i = 0; i < buffer.length; i++) {
            if (buffer[i] >= 0) {
                newBuffer[i] = FastMath.floor(buffer[i] + 0.5);
            } else {
                newBuffer[i] = FastMath.ceil(buffer[i] - 0.5);
            }
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor sigmoid() {
        double[] newBuffer = newBufferOfSameLength();

        Sigmoid sigmoid = new Sigmoid();
        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = sigmoid.value(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor sigmoidInPlace() {

        Sigmoid sigmoid = new Sigmoid();
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = sigmoid.value(buffer[i]);
        }

        return this;
    }

    @Override
    public double product() {
        double result = 1.0;
        for (int i = 0; i < buffer.length; i++) {
            result *= buffer[i];
        }
        return result;
    }

    @Override
    public DoubleTensor slice(int dimension, long index) {

        long[] resultShape = ArrayUtils.remove(shape, dimension);
        long[] resultStride = TensorShape.getRowFirstStride(resultShape);
        double[] newBuffer = new double[Ints.checkedCast(TensorShape.getLength(resultShape))];

        for (int i = 0; i < newBuffer.length; i++) {

            long[] shapeIndices = ArrayUtils.insert(dimension, TensorShape.getShapeIndices(resultShape, resultStride, i), index);

            int j = Ints.checkedCast(TensorShape.getFlatIndex(shape, stride, shapeIndices));

            newBuffer[i] += buffer[j];
        }

        return new JVMDoubleTensor(newBuffer, resultShape);
    }

    public static DoubleTensor concat(int dimension, DoubleTensor... toConcat) {

        long[] concatShape = getConcatResultShape(dimension, toConcat);

        boolean shouldRearrange = dimension != 0;

        if (shouldRearrange) {

            int[] rearrange = shiftDimensionToDimensionZero(dimension, concatShape);

            DoubleTensor[] toConcatOnDimensionZero = new DoubleTensor[toConcat.length];

            for (int i = 0; i < toConcatOnDimensionZero.length; i++) {
                toConcatOnDimensionZero[i] = toConcat[i].permute(rearrange);
            }

            long[] permutedConcatShape = TensorShape.getPermutedResultShapeShape(concatShape, rearrange);
            JVMDoubleTensor concatOnDimZero = concatOnDimensionZero(permutedConcatShape, toConcatOnDimensionZero);

            return concatOnDimZero.permute(invertedPermute(rearrange));
        } else {

            return concatOnDimensionZero(concatShape, toConcat);
        }
    }

    private static JVMDoubleTensor concatOnDimensionZero(long[] concatShape, DoubleTensor... toConcat) {

        double[] concatBuffer = new double[TensorShape.getLengthAsInt(concatShape)];
        int bufferPosition = 0;

        for (int i = 0; i < toConcat.length; i++) {

            double[] cBuffer = getRawBufferIfJVMTensor(toConcat[i]);
            System.arraycopy(cBuffer, 0, concatBuffer, bufferPosition, cBuffer.length);
            bufferPosition += cBuffer.length;
        }

        return new JVMDoubleTensor(concatBuffer, concatShape);
    }

    private static int[] shiftDimensionToDimensionZero(int dimension, long[] shape) {

        int[] rearrange = new int[shape.length];
        rearrange[0] = dimension;
        for (int i = 1; i < rearrange.length; i++) {
            if (i > dimension) {
                rearrange[i] = i;
            } else {
                rearrange[i] = i - 1;
            }
        }
        return rearrange;
    }

    private static long[] getConcatResultShape(int dimension, DoubleTensor... toConcat) {
        Preconditions.checkArgument(toConcat.length > 0);

        DoubleTensor first = toConcat[0];
        long[] firstShape = first.getShape();

        if (firstShape.length == 0 && dimension != 0) {
            throw new IllegalArgumentException("Cannot concat scalars on dimension " + dimension);
        }

        long[] concatShape = firstShape.length == 0 ? new long[]{1} : Arrays.copyOf(firstShape, firstShape.length);

        for (int i = 1; i < toConcat.length; i++) {
            DoubleTensor c = toConcat[i];

            long[] cShape = c.getShape();
            for (int dim = 0; dim < concatShape.length; dim++) {

                if (dim == dimension) {
                    concatShape[dimension] += cShape.length == 0 ? 1 : cShape[dimension];
                } else {
                    if (cShape[dim] != concatShape[dim]) {
                        throw new IllegalArgumentException("Cannot concat shape " + Arrays.toString(cShape));
                    }
                }
            }
        }

        return concatShape;
    }

    @Override
    public double[] asFlatDoubleArray() {
        return bufferCopy();
    }

    private static double[] getRawBufferIfJVMTensor(DoubleTensor tensor) {
        if (tensor instanceof JVMDoubleTensor) {
            return ((JVMDoubleTensor) tensor).buffer;
        } else {
            return tensor.asFlatDoubleArray();
        }
    }

    @Override
    public int[] asFlatIntegerArray() {
        return bufferAsInteger();
    }

    @Override
    public Double[] asFlatArray() {
        Double[] boxedBuffer = new Double[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            boxedBuffer[i] = buffer[i];
        }
        return boxedBuffer;
    }

    /**
     * @param dimension      the dimension to split on
     * @param splitAtIndices the indices that the dimension to split on should be split on
     * @return pieces of the tensor split in the order specified by splitAtIndices. To get
     * pieces that encompasses the entire tensor, the last index in the splitAtIndices must
     * be the length of the dimension being split on.
     * <p>
     * e.g A =
     * [
     * 1, 2, 3, 4, 5, 6
     * 7, 8, 9, 1, 2, 3
     * ]
     * <p>
     * A.split(0, [1]) gives List([1, 2, 3, 4, 5, 6])
     * A.split(0, [1, 2]) gives List([1, 2, 3, 4, 5, 6], [7, 8, 9, 1, 2, 3]
     * <p>
     * A.split(1, [1, 3, 6]) gives
     * List(
     * [1, [2, 3  , [4, 5, 6,
     * 7]  8, 9]    1, 2, 3]
     * )
     */
    @Override
    public List<DoubleTensor> split(int dimension, long... splitAtIndices) {

        long[] shape = getShape();
        dimension = getAbsoluteDimension(dimension, getRank());

        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Invalid dimension to split on " + dimension);
        }

        int[] moveDimToZero = TensorShape.slideDimension(dimension, 0, shape.length);
        int[] moveZeroToDim = TensorShape.slideDimension(0, dimension, shape.length);

        JVMDoubleTensor permutedTensor = (JVMDoubleTensor) this.permute(moveDimToZero).reshape(buffer.length);

        double[] rawBuffer = permutedTensor.buffer;

        List<DoubleTensor> splitTensor = new ArrayList<>();

        long previousSplitAtIndex = 0;
        int rawBufferPosition = 0;
        for (long splitAtIndex : splitAtIndices) {

            long[] subTensorShape = Arrays.copyOf(shape, shape.length);
            long subTensorLengthInDimension = splitAtIndex - previousSplitAtIndex;

            if (subTensorLengthInDimension > shape[dimension] || subTensorLengthInDimension <= 0) {
                throw new IllegalArgumentException("Invalid index to split on " + splitAtIndex + " at " + dimension + " for tensor of shape " + Arrays.toString(shape));
            }

            subTensorShape[dimension] = subTensorLengthInDimension;
            int subTensorLength = Ints.checkedCast(TensorShape.getLength(subTensorShape));

            double[] buffer = new double[subTensorLength];
            System.arraycopy(rawBuffer, rawBufferPosition, buffer, 0, buffer.length);

            long[] subTensorPermutedShape = TensorShape.getPermutedResultShapeShape(subTensorShape, moveDimToZero);
            DoubleTensor subTensor = DoubleTensor.create(buffer, subTensorPermutedShape).permute(moveZeroToDim);
            splitTensor.add(subTensor);

            previousSplitAtIndex = splitAtIndex;
            rawBufferPosition += buffer.length;
        }

        return splitTensor;
    }

    @Override
    public DoubleTensor reciprocalInPlace() {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = 1.0 / buffer[i];
        }
        return this;
    }


    @Override
    public DoubleTensor sqrtInPlace() {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.sqrt(buffer[i]);
        }
        return this;
    }

    @Override
    public DoubleTensor logInPlace() {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.log(buffer[i]);
        }
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
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = Gamma.logGamma(buffer[i]);
        }
        return this;
    }

    @Override
    public DoubleTensor digammaInPlace() {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = Gamma.digamma(buffer[i]);
        }
        return this;
    }

    @Override
    public DoubleTensor sinInPlace() {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.sin(buffer[i]);
        }
        return this;
    }

    @Override
    public DoubleTensor cosInPlace() {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.cos(buffer[i]);
        }
        return this;
    }

    @Override
    public DoubleTensor tanInPlace() {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.tan(buffer[i]);
        }
        return this;
    }

    @Override
    public DoubleTensor atanInPlace() {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.atan(buffer[i]);
        }
        return this;
    }

    @Override
    public DoubleTensor asinInPlace() {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.asin(buffer[i]);
        }
        return this;
    }

    @Override
    public DoubleTensor acosInPlace() {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.acos(buffer[i]);
        }
        return this;
    }

    @Override
    public DoubleTensor expInPlace() {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.exp(buffer[i]);
        }
        return this;
    }

    @Override
    public double min() {
        double result = Double.MAX_VALUE;
        for (int i = 0; i < buffer.length; i++) {
            result = Math.min(result, buffer[i]);
        }
        return result;
    }

    @Override
    public DoubleTensor minInPlace(DoubleTensor that) {
        return broadcastableBinaryDoubleOpInPlace(Math::min, that);
    }

    @Override
    public double max() {
        double result = -Double.MAX_VALUE;
        for (int i = 0; i < buffer.length; i++) {
            result = Math.max(result, buffer[i]);
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
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.ceil(buffer[i]);
        }
        return this;
    }

    @Override
    public DoubleTensor floorInPlace() {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.floor(buffer[i]);
        }
        return this;
    }

    @Override
    public DoubleTensor roundInPlace() {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = FastMath.round(buffer[i]);
        }
        return this;
    }

    @Override
    public DoubleTensor standardizeInPlace() {
        throw new NotImplementedException("");
    }

    @Override
    public DoubleTensor replaceNaNInPlace(double value) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = Double.isNaN(buffer[i]) ? value : buffer[i];
        }
        return this;
    }

    @Override
    public DoubleTensor setAllInPlace(double value) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = value;
        }
        return this;
    }

    @Override
    public BooleanTensor notNaN() {
        boolean[] newBuffer = new boolean[buffer.length];

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = !Double.isNaN(buffer[i]);
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor minusInPlace(double value) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = buffer[i] - value;
        }
        return this;
    }

    @Override
    public DoubleTensor minus(double value) {
        double[] result = new double[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            result[i] = buffer[i] - value;
        }
        return new JVMDoubleTensor(result, shapeCopy());
    }

    @Override
    public DoubleTensor minusInPlace(DoubleTensor that) {
        return broadcastableBinaryDoubleOpInPlace(SUB, that);
    }

    @Override
    public DoubleTensor minus(DoubleTensor that) {
        return broadcastableBinaryDoubleOp(SUB, that);
    }

    @Override
    public DoubleTensor plusInPlace(double value) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = buffer[i] + value;
        }
        return this;
    }

    @Override
    public DoubleTensor plus(double value) {
        double[] result = new double[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            result[i] = buffer[i] + value;
        }
        return new JVMDoubleTensor(result, shapeCopy());
    }

    @Override
    public DoubleTensor plusInPlace(DoubleTensor that) {
        return broadcastableBinaryDoubleOpInPlace(ADD, that);
    }

    @Override
    public DoubleTensor plus(DoubleTensor that) {
        return broadcastableBinaryDoubleOp(ADD, that);
    }

    @Override
    public DoubleTensor timesInPlace(double value) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = buffer[i] * value;
        }
        return this;
    }

    @Override
    public DoubleTensor times(double value) {
        double[] result = new double[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            result[i] = buffer[i] * value;
        }
        return new JVMDoubleTensor(result, shapeCopy());
    }

    @Override
    public DoubleTensor timesInPlace(DoubleTensor that) {
        return broadcastableBinaryDoubleOpInPlace(MUL, that);
    }

    @Override
    public DoubleTensor times(DoubleTensor that) {
        return broadcastableBinaryDoubleOp(MUL, that);
    }

    @Override
    public DoubleTensor divInPlace(double value) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = buffer[i] / value;
        }
        return this;
    }

    @Override
    public DoubleTensor div(double value) {
        double[] result = new double[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            result[i] = buffer[i] / value;
        }
        return new JVMDoubleTensor(result, shapeCopy());
    }

    @Override
    public DoubleTensor divInPlace(DoubleTensor that) {
        return broadcastableBinaryDoubleOpInPlace(DIV, that);
    }

    @Override
    public DoubleTensor div(DoubleTensor that) {
        return broadcastableBinaryDoubleOp(DIV, that);
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


        final double[] rightBuffer = getRawBufferIfJVMTensor(right);
        final long[] rightShape = right.getShape();

        final boolean needsBroadcast = !Arrays.equals(shape, rightShape);

        if (needsBroadcast) {

            //Short circuit for broadcast with scalars
            if (shape.length == 0 || rightShape.length == 0) {

                final double[] outputBuffer;
                final long[] resultShape;
                final long[] resultStride;

                if (shape.length == 0) {
                    outputBuffer = new double[rightBuffer.length];
                    resultShape = Arrays.copyOf(rightShape, rightShape.length);
                    resultStride = Arrays.copyOf(right.getStride(), rightShape.length);
                    scalarLeft(buffer[0], rightBuffer, outputBuffer, op);
                } else {
                    outputBuffer = inPlace ? buffer : new double[buffer.length];
                    resultShape = Arrays.copyOf(shape, shape.length);
                    resultStride = stride;
                    scalarRight(buffer, rightBuffer[0], outputBuffer, op);
                }

                if (inPlace) {
                    buffer = outputBuffer;
                    shape = resultShape;
                    stride = resultStride;

                    return this;
                } else {
                    return new JVMDoubleTensor(outputBuffer, resultShape);
                }

            }

            return broadcastBinaryDoubleOp(this,
                buffer, shape,
                rightBuffer, right.getShape(),
                op, inPlace
            );

        } else {

            return elementwiseBinaryOp(this, buffer, rightBuffer, shape, op, inPlace);
        }

    }

    private long[] getShapeOrPadToRank(long[] shape, int rank) {
        if (shape.length == rank) {
            return shape;
        } else {
            return TensorShape.shapeToDesiredRankByPrependingOnes(shape, rank);
        }
    }

    private JVMDoubleTensor broadcastBinaryDoubleOp(JVMDoubleTensor left, double[] leftBuffer, long[] leftShape,
                                                    double[] rightBuffer, long[] rightShape,
                                                    BiFunction<Double, Double, Double> op,
                                                    boolean inPlace) {

        //implicitly pad lower ranks with 1s. E.g. [3, 3] & [3] -> [3, 3] -> [1, 3]
        int resultRank = Math.max(leftShape.length, rightShape.length);
        long[] paddedLeftShape = getShapeOrPadToRank(leftShape, resultRank);
        long[] paddedLeftStride = TensorShape.getRowFirstStride(paddedLeftShape);

        long[] paddedRightShape = getShapeOrPadToRank(rightShape, resultRank);
        long[] paddedRightStride = TensorShape.getRowFirstStride(paddedRightShape);

        long[] resultShape = Shape.broadcastOutputShape(paddedLeftShape, paddedRightShape);
        boolean resultShapeIsLeftSideShape = Arrays.equals(resultShape, paddedLeftShape);

        final double[] outputBuffer;
        if (!resultShapeIsLeftSideShape) {

            boolean resultShapeIsRightSideShape = Arrays.equals(resultShape, paddedRightShape);

            if (!resultShapeIsRightSideShape) {
                throw new IllegalArgumentException(
                    "Broadcasting of shape " + Arrays.toString(paddedLeftShape) + " and " + Arrays.toString(paddedRightShape) + " not supported."
                );
            }

            outputBuffer = new double[Ints.checkedCast(TensorShape.getLength(resultShape))];

        } else {
            outputBuffer = inPlace ? leftBuffer : new double[leftBuffer.length];
        }

        //Allow broadcasting from left and right
        if (paddedLeftShape.length > paddedRightShape.length || leftBuffer.length > rightBuffer.length) {
            //e.g. [2, 2] * [1, 2]
            broadcastFromRight(leftBuffer, paddedLeftShape, paddedLeftStride, rightBuffer, paddedRightShape, paddedRightStride, outputBuffer, op);
        } else {
            //e.g. [2] / [2, 2]
            broadcastFromLeft(leftBuffer, paddedLeftShape, paddedLeftStride, rightBuffer, paddedRightShape, paddedRightStride, outputBuffer, op);
        }

        if (inPlace) {
            left.buffer = outputBuffer;
            left.shape = resultShape;

            //TODO: get rid of the need to do this!
            left.stride = TensorShape.getRowFirstStride(resultShape);

            return left;
        } else {
            return new JVMDoubleTensor(outputBuffer, resultShape);
        }
    }

    private JVMDoubleTensor elementwiseBinaryOp(JVMDoubleTensor left, double[] leftBuffer, double[] rightBuffer, long[] shape,
                                                BiFunction<Double, Double, Double> op,
                                                boolean inPlace) {

        final double[] outputBuffer = inPlace ? leftBuffer : new double[leftBuffer.length];

        for (int i = 0; i < outputBuffer.length; i++) {
            outputBuffer[i] = op.apply(leftBuffer[i], rightBuffer[i]);
        }

        if (inPlace) {
            left.buffer = outputBuffer;
            return left;
        } else {
            return new JVMDoubleTensor(outputBuffer, shape);
        }
    }

    @Override
    public FlattenedView<Double> getFlattenedView() {
        return new JVMDoubleFlattenedView();
    }

    private class JVMDoubleFlattenedView implements FlattenedView<Double> {

        @Override
        public long size() {
            return buffer.length;
        }

        @Override
        public Double get(long index) {
            return buffer[Ints.checkedCast(index)];
        }

        @Override
        public Double getOrScalar(long index) {
            if (buffer.length == 1) {
                return get(0);
            } else {
                return get(index);
            }
        }

        @Override
        public void set(long index, Double value) {
            buffer[Ints.checkedCast(index)] = value;
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        JVMDoubleTensor that = (JVMDoubleTensor) o;
        return Arrays.equals(shape, that.shape) &&
            Arrays.equals(buffer, that.buffer);
    }

    @Override
    public String toString() {
        return "JVMDoubleTensor{" +
            "shape=" + Arrays.toString(shape) +
            ", buffer=" + Arrays.toString(buffer) +
            '}';
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(buffer);
        return result;
    }

}
