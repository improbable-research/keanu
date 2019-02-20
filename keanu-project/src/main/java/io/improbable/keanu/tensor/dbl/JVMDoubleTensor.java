package io.improbable.keanu.tensor.dbl;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.shape.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;
import static io.improbable.keanu.tensor.TensorShape.invertedPermute;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.ADD;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.DIV;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.GTE_MASK;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.GT_MASK;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.LTE_MASK;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.LT_MASK;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.MUL;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.BroadcastableDoubleOperation.SUB;
import static io.improbable.keanu.tensor.dbl.JVMDoubleTensorBroadcast.opWithAutoBroadcast;
import static java.util.Arrays.copyOf;

public class JVMDoubleTensor extends DoubleTensor {

    private long[] shape;
    private long[] stride;
    private double[] buffer;

    public JVMDoubleTensor(double value) {
        this.shape = new long[0];
        this.stride = new long[0];
        this.buffer = new double[]{value};
    }

    private JVMDoubleTensor(double[] data, long[] shape) {
        this.shape = shape;
        this.stride = TensorShape.getRowFirstStride(shape);
        this.buffer = data;
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
        long length = TensorShape.getLength(shape);
        double[] buffer = new double[Ints.checkedCast(length)];
        Arrays.fill(buffer, value);
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
        int steps = (int) Math.ceil((end - start) / stepSize);
        double[] buffer = new double[steps];

        double position = start;
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = position;
            position += stepSize;
        }

        return new JVMDoubleTensor(buffer, new long[]{buffer.length});
    }

    public static JVMDoubleTensor linspace(double start, double end, int numberOfPoints) {

        double stepSize = (end - start) / numberOfPoints;

        double[] buffer = new double[numberOfPoints];

        double position = start;
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = position;
            position += stepSize;
        }

        return new JVMDoubleTensor(buffer, new long[]{buffer.length});
    }

    private double[] newBuffer() {
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

    private void checkElementwiseShapeMatch(long[] otherShape) {
        if (!Arrays.equals(shape, otherShape)) {
            throw new IllegalArgumentException(
                "Broadcast not supported for shape " +
                    Arrays.toString(shape) +
                    " and " +
                    Arrays.toString(otherShape)
            );
        }
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
        return buffer.length > 0 ? buffer.length : 1;
    }

    @Override
    public Double getValue(long... index) {
        long flatIndex = TensorShape.getFlatIndex(shape, stride, index);
        return buffer[Ints.checkedCast(flatIndex)];
    }

    @Override
    public DoubleTensor setValue(Double value, long... index) {
        long flatIndex = TensorShape.getFlatIndex(shape, stride, index);
        buffer[Ints.checkedCast(flatIndex)] = value;
        return this;
    }

    @Override
    public Double scalar() {
        return buffer[0];
    }

    @Override
    public DoubleTensor reshape(long... newShape) {

        long newLength = 1;
        int negativeDimension = -1;

        for (int i = 0; i < newShape.length; i++) {

            long dimILength = newShape[i];
            if (dimILength > 0) {
                newLength *= dimILength;
            } else if (dimILength < 0) {
                if (negativeDimension >= 0) {
                    throw new IllegalArgumentException("Cannot reshape " + Arrays.toString(shape) + " to " + Arrays.toString(newShape));
                }
                negativeDimension = i;
            }
        }

        if (newLength != buffer.length) {
            if (negativeDimension < 0) {
                throw new IllegalArgumentException("Cannot reshape " + Arrays.toString(shape) + " to " + Arrays.toString(newShape));
            } else {
                newShape[negativeDimension] = buffer.length / newLength;
            }
        }


        return new JVMDoubleTensor(copyOf(buffer, buffer.length), copyOf(newShape, newShape.length));
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

        long[] resultShape = TensorShape.getPermutedResultShapeShape(shape, rearrange);
        long[] resultStride = TensorShape.getRowFirstStride(resultShape);
        double[] newBuffer = newBuffer();

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
    public DoubleTensor matrixInverse() {
        return fromApacheRealMatrix(MatrixUtils.inverse(asApacheRealMatrix(this)));
    }

    @Override
    public DoubleTensor matrixMultiply(DoubleTensor that) {

        RealMatrix thisMatrix = asApacheRealMatrix(this);
        RealMatrix thatMatrix = asApacheRealMatrix(that);

        return fromApacheRealMatrix(thisMatrix.multiply(thatMatrix));
    }

    private static RealMatrix asApacheRealMatrix(DoubleTensor matrix) {
        long[] shape = matrix.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("Cannot convert tensor of shape " + Arrays.toString(shape) + " to a matrix.");
        }

        BlockRealMatrix out = new BlockRealMatrix((int) shape[0], (int) shape[1]);
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                double value = matrix.getValue(i, j);
                out.setEntry(i, j, value);
            }
        }
        return out;
    }

    private static JVMDoubleTensor fromApacheRealMatrix(RealMatrix realMatrix) {
        double[][] data = realMatrix.getData();
        double[] flatData = new double[realMatrix.getRowDimension() * realMatrix.getColumnDimension()];

        int rows = realMatrix.getRowDimension();
        int cols = realMatrix.getColumnDimension();
        for (int r = 0; r < rows; r++) {
            System.arraycopy(data[r], 0, flatData, r * cols, cols);
        }

        return new JVMDoubleTensor(flatData, new long[]{rows, cols});
    }

    @Override
    public DoubleTensor tensorMultiply(DoubleTensor value, int[] dimsLeft, int[] dimsRight) {
        throw new NotImplementedException("");
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
        throw new NotImplementedException("");
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

        double[] newBuffer = newBuffer();

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
        return broadcastableOp(GT_MASK, greaterThanThis);
    }

    @Override
    public DoubleTensor getGreaterThanOrEqualToMask(DoubleTensor greaterThanThis) {
        return broadcastableOp(GTE_MASK, greaterThanThis);
    }

    @Override
    public DoubleTensor getLessThanMask(DoubleTensor lessThanThis) {
        return broadcastableOp(LT_MASK, lessThanThis);
    }

    @Override
    public DoubleTensor getLessThanOrEqualToMask(DoubleTensor lessThanThis) {
        return broadcastableOp(LTE_MASK, lessThanThis);
    }

    @Override
    public DoubleTensor setWithMaskInPlace(DoubleTensor mask, Double value) {
        checkElementwiseShapeMatch(mask.getShape());

        double[] maskBuffer = mask.asFlatDoubleArray();

        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = maskBuffer[i] == 1.0 ? value : buffer[i];
        }

        return this;
    }

    @Override
    public DoubleTensor setWithMask(DoubleTensor mask, Double value) {
        checkElementwiseShapeMatch(mask.getShape());

        double[] newBuffer = new double[buffer.length];
        double[] maskBuffer = mask.asFlatDoubleArray();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = maskBuffer[i] == 1.0 ? value : buffer[i];
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public BooleanTensor lessThan(DoubleTensor that) {
        boolean[] newBuffer = new boolean[buffer.length];
        double[] thatBuffer = that.asFlatDoubleArray();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = buffer[i] < thatBuffer[i];
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public BooleanTensor lessThanOrEqual(DoubleTensor that) {
        boolean[] newBuffer = new boolean[buffer.length];
        double[] thatBuffer = that.asFlatDoubleArray();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = buffer[i] <= thatBuffer[i];
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public BooleanTensor greaterThan(DoubleTensor that) {

        boolean[] newBuffer = new boolean[buffer.length];
        double[] thatBuffer = that.asFlatDoubleArray();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = buffer[i] > thatBuffer[i];
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
    }

    @Override
    public BooleanTensor greaterThanOrEqual(DoubleTensor that) {
        boolean[] newBuffer = new boolean[buffer.length];
        double[] thatBuffer = that.asFlatDoubleArray();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = buffer[i] >= thatBuffer[i];
        }

        return BooleanTensor.create(newBuffer, shapeCopy());
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
        return broadcastableOpInPlace(FastMath::pow, exponent);
    }

    @Override
    public DoubleTensor pow(DoubleTensor exponent) {
        return broadcastableOp(FastMath::pow, exponent);
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

        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.pow(buffer[i], exponent);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor sqrt() {
        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.sqrt(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor log() {
        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.log(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor safeLogTimes(DoubleTensor y) {
        throw new NotImplementedException("");
    }

    @Override
    public DoubleTensor logGamma() {

        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = Gamma.logGamma(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor digamma() {
        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = Gamma.digamma(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor sin() {
        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.sin(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor cos() {
        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.cos(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor tan() {
        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.tan(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor atan() {
        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.atan(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor atan2(double y) {
        double[] newBuffer = newBuffer();

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
        return broadcastableOpInPlace((left, right) -> FastMath.atan2(right, left), y);
    }

    @Override
    public DoubleTensor atan2(DoubleTensor y) {
        return broadcastableOp((left, right) -> FastMath.atan2(right, left), y);
    }

    @Override
    public DoubleTensor asin() {
        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.asin(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor acos() {
        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.acos(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor exp() {
        double[] newBuffer = newBuffer();

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
        return 0;
    }

    @Override
    public boolean equalsWithinEpsilon(DoubleTensor other, double epsilon) {
        if (!Arrays.equals(shape, other.getShape())) {
            return false;
        }

        double[] otherBuffer = other.asFlatDoubleArray();

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
        double[] newBuffer = newBuffer();

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
        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.ceil(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor floor() {
        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.floor(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor round() {
        double[] newBuffer = newBuffer();

        for (int i = 0; i < buffer.length; i++) {
            newBuffer[i] = FastMath.round(buffer[i]);
        }

        return new JVMDoubleTensor(newBuffer, shapeCopy());
    }

    @Override
    public DoubleTensor sigmoid() {
        double[] newBuffer = newBuffer();

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
    public DoubleTensor choleskyDecomposition() {
//        return fromApacheRealMatrix(new CholeskyDecomposition(asApacheRealMatrix(this)).getL());
        throw new NotImplementedException("");
    }

    @Override
    public double determinant() {
        return 0;
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

            double[] cBuffer = toConcat[i].asFlatDoubleArray();
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

        DoubleTensor permutedTensor = this.permute(moveDimToZero).reshape(buffer.length);

        double[] rawBuffer = permutedTensor.asFlatDoubleArray();

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
        throw new NotImplementedException("");
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
        return broadcastableOpInPlace(Math::min, that);
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
        return broadcastableOpInPlace(Math::max, that);
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
        return broadcastableOpInPlace(SUB, that);
    }

    @Override
    public DoubleTensor minus(DoubleTensor that) {
        return broadcastableOp(SUB, that);
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
        return broadcastableOpInPlace(ADD, that);
    }

    @Override
    public DoubleTensor plus(DoubleTensor that) {
        return broadcastableOp(ADD, that);
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
        return broadcastableOpInPlace(MUL, that);
    }

    @Override
    public DoubleTensor times(DoubleTensor that) {
        return broadcastableOp(MUL, that);
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
        return broadcastableOpInPlace(DIV, that);
    }

    @Override
    public DoubleTensor div(DoubleTensor that) {
        return broadcastableOp(DIV, that);
    }

    private DoubleTensor broadcastableOp(BiFunction<Double, Double, Double> op, DoubleTensor that) {

        long[] resultShape = Shape.broadcastOutputShape(shape, that.getShape());

        boolean resultShapeIsThisShape = Arrays.equals(resultShape, shape);
        boolean resultShapeIsThatShape = Arrays.equals(resultShape, that.getShape());

        if (!resultShapeIsThisShape && !resultShapeIsThatShape) {
            throw new IllegalArgumentException(
                "Broadcasting of shape " + Arrays.toString(shape) + " and " + Arrays.toString(that.getShape()) + " not supported."
            );
        }

        double[] outputBuffer;
        if (resultShapeIsThisShape) {
            outputBuffer = newBuffer();
        } else {
            outputBuffer = new double[Ints.checkedCast(TensorShape.getLength(resultShape))];
        }

        double[] result = opWithAutoBroadcast(buffer, shape, stride, that, outputBuffer, op);

        return new JVMDoubleTensor(result, resultShape);
    }

    private DoubleTensor broadcastableOpInPlace(BiFunction<Double, Double, Double> op, DoubleTensor that) {

        long[] resultShape = Shape.broadcastOutputShape(shape, that.getShape());

        boolean resultShapeIsThisShape = Arrays.equals(resultShape, shape);
        boolean resultShapeIsThatShape = Arrays.equals(resultShape, that.getShape());

        if (!resultShapeIsThisShape && !resultShapeIsThatShape) {
            throw new IllegalArgumentException(
                "Broadcasting of shape " + Arrays.toString(shape) + " and " + Arrays.toString(that.getShape()) + " not supported."
            );
        }

        double[] outputBuffer;
        if (resultShapeIsThisShape) {
            outputBuffer = buffer;
        } else {
            outputBuffer = new double[Ints.checkedCast(TensorShape.getLength(resultShape))];
        }

        this.buffer = opWithAutoBroadcast(buffer, shape, stride, that, outputBuffer, op);

        this.shape = resultShape;

        return this;
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
