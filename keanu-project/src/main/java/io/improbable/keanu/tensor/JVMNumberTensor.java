package io.improbable.keanu.tensor;

import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.buffer.JVMBuffer;
import io.improbable.keanu.tensor.buffer.PrimitiveNumberWrapper;
import org.apache.commons.lang3.ArrayUtils;

import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

import static io.improbable.keanu.tensor.TensorShape.dimensionRange;
import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getLengthAsInt;
import static io.improbable.keanu.tensor.TensorShape.getReductionResultShape;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static io.improbable.keanu.tensor.TensorShape.incrementIndexByShape;
import static io.improbable.keanu.tensor.TensorShape.setToAbsoluteDimensions;

public abstract class JVMNumberTensor<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, B extends PrimitiveNumberWrapper<T, B>> extends JVMTensor<T, TENSOR, B> implements NumberTensor<T, TENSOR> {

    protected JVMNumberTensor(B buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

    @Override
    public T sum() {
        return buffer.sum();
    }

    @Override
    public T product() {
        return buffer.product();
    }

    @Override
    public TENSOR sum(int... overDimensions) {
        return reduceOverDimensions(
            PrimitiveNumberWrapper::plus,
            JVMBuffer.PrimitiveNumberWrapperFactory::zeroes,
            PrimitiveNumberWrapper::sum,
            overDimensions
        );
    }

    @Override
    public TENSOR product(int... overDimensions) {
        return reduceOverDimensions(
            PrimitiveNumberWrapper::times,
            JVMBuffer.PrimitiveNumberWrapperFactory::ones,
            PrimitiveNumberWrapper::product,
            overDimensions
        );
    }

    interface BufferOp<T extends Number, B extends PrimitiveNumberWrapper<T, B>> {
        void apply(PrimitiveNumberWrapper<T, B> buffer, long j, T value);
    }

    private TENSOR reduceOverDimensions(BufferOp<T, B> combine,
                                        BiFunction<JVMBuffer.PrimitiveNumberWrapperFactory<T, B>, Integer, B> init,
                                        Function<B, T> totalReduction,
                                        int... overDimensions) {

        setToAbsoluteDimensions(this.shape.length, overDimensions);

        if (this.isScalar() || overDimensions.length == 0) {
            return duplicate();
        } else if (this.isVector()) {
            B aNew = getFactory().createNew(totalReduction.apply(buffer));
            return create(aNew, new long[0], new long[0]);
        }

        long[] resultShape = getReductionResultShape(shape, overDimensions);
        long[] resultStride = getRowFirstStride(resultShape);
        B newBuffer = init.apply(getFactory(), getLengthAsInt(resultShape));

        for (int i = 0; i < buffer.getLength(); i++) {

            long[] shapeIndices = ArrayUtils.removeAll(TensorShape.getShapeIndices(shape, stride, i), overDimensions);

            int j = Ints.checkedCast(getFlatIndex(resultShape, resultStride, shapeIndices));

            combine.apply(newBuffer, j, buffer.get(i));
        }

        return create(newBuffer, resultShape, resultStride);
    }

    @Override
    public TENSOR cumSumInPlace(int requestedDimension) {
        return cumulativeInPlace(
            PrimitiveNumberWrapper::plus,
            requestedDimension
        );
    }

    @Override
    public TENSOR cumProdInPlace(int requestedDimension) {
        return cumulativeInPlace(
            PrimitiveNumberWrapper::times,
            requestedDimension);
    }

    private TENSOR cumulativeInPlace(BufferOp<T, B> combine, int requestedDimension) {

        final int dimension = requestedDimension >= 0 ? requestedDimension : requestedDimension + shape.length;
        TensorShapeValidation.checkDimensionExistsInShape(dimension, shape);
        final int[] dimensionOrder = ArrayUtils.remove(dimensionRange(0, shape.length), dimension);
        long[] index = new long[shape.length];

        do {

            T result = null;
            for (int i = 0; i < shape[dimension]; i++) {

                index[dimension] = i;

                long j = getFlatIndex(shape, stride, index);

                if (i > 0) {
                    combine.apply(buffer, j, result);
                }

                result = buffer.get(j);
            }

        } while (incrementIndexByShape(shape, index, dimensionOrder));

        return set(buffer, shape, stride);
    }

    @Override
    protected abstract JVMBuffer.PrimitiveNumberWrapperFactory<T, B> getFactory();

}
