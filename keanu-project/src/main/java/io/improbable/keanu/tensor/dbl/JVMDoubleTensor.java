package io.improbable.keanu.tensor.dbl;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.NumberScalarOperations;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.JVMBooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.jvm.JVMFloatingPointTensor;
import io.improbable.keanu.tensor.jvm.JVMNumberTensor;
import io.improbable.keanu.tensor.jvm.JVMTensor;
import io.improbable.keanu.tensor.jvm.ResultWrapper;
import io.improbable.keanu.tensor.lng.JVMLongTensor;
import io.improbable.keanu.tensor.lng.LongTensor;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;
import java.util.stream.Collectors;

import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static io.improbable.keanu.tensor.dbl.BroadcastableDoubleOperations.LOG_ADD_EXP;
import static io.improbable.keanu.tensor.dbl.BroadcastableDoubleOperations.LOG_ADD_EXP2;
import static io.improbable.keanu.tensor.dbl.BroadcastableDoubleOperations.SAFE_LOG_TIMES;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dgetrf;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dgetri;
import static io.improbable.keanu.tensor.dbl.KeanuLapack.dpotrf;
import static org.bytedeco.openblas.global.openblas.CblasNoTrans;
import static org.bytedeco.openblas.global.openblas.CblasRowMajor;
import static org.bytedeco.openblas.global.openblas.cblas_dgemm;

public class JVMDoubleTensor extends JVMFloatingPointTensor<Double, DoubleTensor, DoubleBuffer.PrimitiveDoubleWrapper> implements DoubleTensor {

    private static final DoubleBuffer.DoubleArrayWrapperFactory factory = new DoubleBuffer.DoubleArrayWrapperFactory();

    private JVMDoubleTensor(DoubleBuffer.PrimitiveDoubleWrapper buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

    private JVMDoubleTensor(DoubleBuffer.PrimitiveDoubleWrapper buffer, long[] shape) {
        super(buffer, shape, getRowFirstStride(shape));
    }

    private JVMDoubleTensor(ResultWrapper<Double, DoubleBuffer.PrimitiveDoubleWrapper> resultWrapper) {
        this(resultWrapper.outputBuffer, resultWrapper.outputShape, resultWrapper.outputStride);
    }

    private JVMDoubleTensor(double[] data, long[] shape, long[] stride) {
        this(factory.create(data), shape, stride);
    }

    private JVMDoubleTensor(double[] data, long[] shape) {
        this(factory.create(data), shape);
    }

    private JVMDoubleTensor(double value) {
        super(new DoubleBuffer.DoubleWrapper(value), new long[0], new long[0]);
    }

    @Override
    protected DoubleTensor create(DoubleBuffer.PrimitiveDoubleWrapper buffer, long[] shape, long[] stride) {
        return new JVMDoubleTensor(buffer, shape, stride);
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
            return new JVMDoubleTensor(new DoubleBuffer.DoubleWrapper(value), shape);
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

    @Override
    protected DoubleTensor set(DoubleBuffer.PrimitiveDoubleWrapper buffer, long[] shape, long[] stride) {
        this.buffer = buffer;
        this.shape = shape;
        this.stride = stride;
        return this;
    }

    @Override
    protected DoubleBuffer.DoubleArrayWrapperFactory getFactory() {
        return factory;
    }

    @Override
    protected NumberScalarOperations<Double> getOperations() {
        return DoubleScalarOperations.INSTANCE;
    }

    @Override
    protected JVMTensor<Double, DoubleTensor, DoubleBuffer.PrimitiveDoubleWrapper> getAsJVMTensor(DoubleTensor that) {
        return asJVM(that);
    }

    private static JVMNumberTensor<Double, DoubleTensor, DoubleBuffer.PrimitiveDoubleWrapper> asJVM(DoubleTensor that) {
        if (that instanceof JVMDoubleTensor) {
            return ((JVMDoubleTensor) that);
        } else {
            return new JVMDoubleTensor(factory.create(that.asFlatDoubleArray()), that.getShape(), that.getStride());
        }
    }

    @Override
    public BooleanTensor toBoolean() {
        return new JVMBooleanTensor(buffer.equal(1.0), getShape(), getStride());
    }

    @Override
    public DoubleTensor toDouble() {
        return this;
    }

    @Override
    public IntegerTensor toInteger() {
        return IntegerTensor.create(buffer.asIntegerArray(), getShape());
    }

    @Override
    public LongTensor toLong() {
        return JVMLongTensor.create(buffer.asLongArray(), getShape());
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

        return new JVMDoubleTensor(newBuffer, getShape(), getStride());
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
    public DoubleTensor matrixDeterminant() {

        final int m = Ints.checkedCast(shape[0]);
        final int n = Ints.checkedCast(shape[1]);
        final double[] newBuffer = buffer.copy().asDoubleArray();
        final int[] ipiv = new int[newBuffer.length];

        final int factorizationResult = dgetrf(m, n, newBuffer, ipiv);

        if (factorizationResult < 0) {
            throw new IllegalStateException("Matrix factorization failed");
        } else if (factorizationResult > 0) {
            return new JVMDoubleTensor(0.0);
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

        return new JVMDoubleTensor(detU * detp);
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

        return new JVMDoubleTensor(newBuffer, getShape(), getStride());
    }

    @Override
    public DoubleTensor matrixMultiply(DoubleTensor that) {

        final long[] thatShape = that.getShape();
        TensorShapeValidation.getMatrixMultiplicationResultingShape(shape, thatShape);

        //C = alpha*A*B + beta*C
        //(M,N) = (M,k)(k,N) + (M,N)
        final double[] A = buffer.asDoubleArray();
        final double[] B = getAsJVMTensor(that).getBuffer().asDoubleArray();
        final double[] C = new double[Ints.checkedCast(this.shape[0] * thatShape[1])];

        final int N = Ints.checkedCast(thatShape[1]);
        final int M = Ints.checkedCast(this.shape[0]);
        final int K = Ints.checkedCast(this.shape[1]);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);

        return new JVMDoubleTensor(C, new long[]{this.shape[0], thatShape[1]});
    }

    @Override
    public IntegerTensor nanArgMax(int axis) {
        return argCompare((value, max) -> Double.isNaN(max) || !Double.isNaN(value) && value > max, axis);
    }

    @Override
    public IntegerTensor nanArgMax() {
        return IntegerTensor.scalar(argCompare((value, max) -> Double.isNaN(max) || !Double.isNaN(value) && value > max));
    }

    @Override
    public IntegerTensor nanArgMin(int axis) {
        return argCompare((value, min) -> Double.isNaN(min) || !Double.isNaN(value) && value < min, axis);
    }

    @Override
    public IntegerTensor nanArgMin() {
        return IntegerTensor.scalar(argCompare((value, min) -> Double.isNaN(min) || !Double.isNaN(value) && value < min));
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
    public DoubleTensor signInPlace() {
        buffer.apply(Math::signum);
        return this;
    }

    @Override
    public BooleanTensor equalsWithinEpsilon(DoubleTensor that, Double epsilon) {
        return broadcastableBinaryOpToBooleanWithAutoBroadcast(
            (l, r) -> Math.abs(l - r) <= epsilon, getAsJVMTensor(that)
        );
    }

    @Override
    public DoubleTensor setWithMaskInPlace(DoubleTensor mask, Double value) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace((l, r) -> r == 1L ? value : l, getAsJVMTensor(mask));
    }

    @Override
    public DoubleTensor atan2InPlace(Double y) {
        buffer.applyLeft(FastMath::atan2, y);
        return this;
    }

    @Override
    public DoubleTensor atan2InPlace(DoubleTensor y) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace((left, right) -> FastMath.atan2(right, left), getAsJVMTensor(y));
    }

    @Override
    public DoubleTensor mean() {
        return new JVMDoubleTensor(buffer.sum() / buffer.getLength());
    }

    @Override
    public DoubleTensor mean(int... overDimensions) {
        final long length = TensorShape.getLength(shape, overDimensions);
        return sum(overDimensions).divInPlace((double) length);
    }

    @Override
    public DoubleTensor standardDeviation() {

        SummaryStatistics stats = new SummaryStatistics();
        for (int i = 0; i < buffer.getLength(); i++) {
            stats.addValue(buffer.get(i));
        }

        return new JVMDoubleTensor(stats.getStandardDeviation());
    }

    private static final Sigmoid sigmoid = new Sigmoid();

    @Override
    public DoubleTensor sigmoidInPlace() {
        buffer.apply(sigmoid::value);
        return this;
    }

    public static DoubleTensor concat(int dimension, DoubleTensor... toConcat) {
        return new JVMDoubleTensor(
            JVMTensor.concat(factory, toConcat, dimension,
                Arrays.stream(toConcat)
                    .map(tensor -> asJVM(tensor).getBuffer())
                    .collect(Collectors.toList())
            ));
    }

    @Override
    public double[] asFlatDoubleArray() {
        return buffer.copy().asDoubleArray();
    }

    @Override
    public int[] asFlatIntegerArray() {
        return buffer.asIntegerArray();
    }

    @Override
    public long[] asFlatLongArray() {
        return buffer.asLongArray();
    }

    @Override
    public Double[] asFlatArray() {
        return buffer.asArray();
    }

    @Override
    public DoubleTensor reciprocalInPlace() {
        buffer.reverseDiv(1.0);
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
        return broadcastableBinaryOpWithAutoBroadcastInPlace(SAFE_LOG_TIMES, getAsJVMTensor(y));
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
    public DoubleTensor trigammaInPlace() {
        buffer.apply(Gamma::trigamma);
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
    public DoubleTensor sinhInPlace() {
        buffer.apply(FastMath::sinh);
        return this;
    }

    @Override
    public DoubleTensor coshInPlace() {
        buffer.apply(FastMath::cosh);
        return this;
    }

    @Override
    public DoubleTensor tanhInPlace() {
        buffer.apply(FastMath::tanh);
        return this;
    }

    @Override
    public DoubleTensor asinhInPlace() {
        buffer.apply(FastMath::asinh);
        return this;
    }

    @Override
    public DoubleTensor acoshInPlace() {
        buffer.apply(FastMath::acosh);
        return this;
    }

    @Override
    public DoubleTensor atanhInPlace() {
        buffer.apply(FastMath::atanh);
        return this;
    }

    @Override
    public DoubleTensor expInPlace() {
        buffer.apply(FastMath::exp);
        return this;
    }

    @Override
    public DoubleTensor logAddExp2InPlace(DoubleTensor that) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace(LOG_ADD_EXP2, getAsJVMTensor(that));
    }

    @Override
    public DoubleTensor logAddExpInPlace(DoubleTensor that) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace(LOG_ADD_EXP, getAsJVMTensor(that));
    }

    @Override
    public DoubleTensor log1pInPlace() {
        buffer.apply(FastMath::log1p);
        return this;
    }

    @Override
    public DoubleTensor log2InPlace() {
        buffer.apply(v -> FastMath.log(v) / FastMath.log(2));
        return this;
    }

    @Override
    public DoubleTensor log10InPlace() {
        buffer.apply(FastMath::log10);
        return this;
    }

    @Override
    public DoubleTensor exp2InPlace() {
        buffer.apply(v -> FastMath.pow(2, v));
        return this;
    }

    @Override
    public DoubleTensor expM1InPlace() {
        buffer.apply(FastMath::expm1);
        return this;
    }

    @Override
    public DoubleTensor min() {
        double result = Double.MAX_VALUE;
        for (int i = 0; i < buffer.getLength(); i++) {
            result = Math.min(result, buffer.get(i));
        }
        return new JVMDoubleTensor(result);
    }

    @Override
    public DoubleTensor minInPlace(DoubleTensor that) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace(Math::min, getAsJVMTensor(that));
    }

    @Override
    public DoubleTensor max() {
        double result = -Double.MAX_VALUE;
        for (int i = 0; i < buffer.getLength(); i++) {
            result = Math.max(result, buffer.get(i));
        }
        return new JVMDoubleTensor(result);
    }

    @Override
    public DoubleTensor maxInPlace(DoubleTensor that) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace(Math::max, getAsJVMTensor(that));
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
        return this.minusInPlace(mean()).divInPlace(standardDeviation());
    }

    @Override
    public DoubleTensor replaceNaNInPlace(Double value) {
        for (int i = 0; i < buffer.getLength(); i++) {
            buffer.set(Double.isNaN(buffer.get(i)) ? value : buffer.get(i), i);
        }
        return this;
    }

    @Override
    public BooleanTensor notNaN() {
        return isApply(v -> !Double.isNaN(v));
    }

    @Override
    public BooleanTensor isNaN() {
        return isApply(v -> Double.isNaN(v));
    }

    @Override
    public BooleanTensor isFinite() {
        return isApply(Double::isFinite);
    }

    @Override
    public BooleanTensor isInfinite() {
        return isApply(v -> Double.isInfinite(v));
    }

    @Override
    public BooleanTensor isNegativeInfinity() {
        return isApply(v -> v == Double.NEGATIVE_INFINITY);
    }

    @Override
    public BooleanTensor isPositiveInfinity() {
        return isApply(v -> v == Double.POSITIVE_INFINITY);
    }

}
