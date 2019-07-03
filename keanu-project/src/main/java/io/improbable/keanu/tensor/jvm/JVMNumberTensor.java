package io.improbable.keanu.tensor.jvm;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.buffer.JVMBuffer;
import io.improbable.keanu.tensor.buffer.PrimitiveNumberWrapper;
import org.apache.commons.lang3.ArrayUtils;

import java.util.function.BiFunction;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShape.dimensionRange;
import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
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

    /**
     * This method works by iterating over the entire buffer and calculating which index in the result buffer
     * it should be combined with using the combine function.
     *
     * @param combine        combines two numbers and returns a single number
     * @param init           a function that returns a buffer initialized with a number suitable for the combine function
     *                       to start. E.g. for a product reduction this would initialize a buffer to ones such that the
     *                       first reduction is called with 1 * element 0.
     * @param totalReduction A function that returns the reduction result of the entire buffer. This is in some cases
     *                       possible and can be much more performant than the permutation walk done otherwise.
     * @param overDimensions The dimensions to reduce over.
     * @return a tensor with a shape with overDimensions dropped and with values reduced.
     */
    private TENSOR reduceOverDimensions(BufferOp<T, B> combine,
                                        BiFunction<JVMBuffer.PrimitiveNumberWrapperFactory<T, B>, Long, B> init,
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
        B newBuffer = init.apply(getFactory(), TensorShape.getLength(resultShape));

        for (int i = 0; i < buffer.getLength(); i++) {

            long[] shapeIndices = ArrayUtils.removeAll(TensorShape.getShapeIndices(shape, stride, i), overDimensions);

            long j = getFlatIndex(resultShape, resultStride, shapeIndices);

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

    /**
     * A cumulative reduce of the buffer. E.g. for summation of [1,2,3] this would return
     * [1, 3, 6] where the sum is applied as [1, 1+2, 1+2+3].
     *
     * @param combine            combines two numbers and returns a single number
     * @param requestedDimension cumulative operation over this dimension
     * @return a tensor of the same shape with the combine operation applied to the requestedDimension
     */
    private TENSOR cumulativeInPlace(BufferOp<T, B> combine, int requestedDimension) {

        final int dimension = requestedDimension >= 0 ? requestedDimension : requestedDimension + shape.length;
        TensorShapeValidation.checkDimensionExistsInShape(dimension, shape);
        final int[] dimensionOrder = ArrayUtils.remove(dimensionRange(0, shape.length), dimension);
        long[] index = new long[shape.length];

        do {

            T result = null;
            for (long i = 0; i < shape[dimension]; i++) {

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
