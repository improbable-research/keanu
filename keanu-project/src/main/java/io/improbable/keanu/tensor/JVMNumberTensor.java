package io.improbable.keanu.tensor;

import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.buffer.PrimitiveNumberWrapper;
import org.apache.commons.lang3.ArrayUtils;

import static io.improbable.keanu.tensor.TensorShape.dimensionRange;
import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static io.improbable.keanu.tensor.TensorShape.getSummationResultShape;
import static io.improbable.keanu.tensor.TensorShape.incrementIndexByShape;

public abstract class JVMNumberTensor<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, B extends PrimitiveNumberWrapper<T, B>> extends JVMTensor<T, TENSOR, B> implements NumberTensor<T, TENSOR> {

    protected JVMNumberTensor(B buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

    @Override
    public T sum() {
        return buffer.sum();
    }

    @Override
    public TENSOR sum(int... overDimensions) {

        overDimensions = TensorShape.getAbsoluteDimensions(this.shape.length, overDimensions);

        if (this.isScalar() || overDimensions.length == 0) {
            return duplicate();
        } else if (this.isVector()) {
            B aNew = getFactory().createNew(1);
            aNew.set(buffer.sum(), 0);
            return create(aNew, new long[0], new long[0]);
        }

        long[] resultShape = getSummationResultShape(shape, overDimensions);
        long[] resultStride = getRowFirstStride(resultShape);
        B newBuffer = getFactory().createNew(TensorShape.getLengthAsInt(resultShape));

        for (int i = 0; i < buffer.getLength(); i++) {

            long[] shapeIndices = ArrayUtils.removeAll(TensorShape.getShapeIndices(shape, stride, i), overDimensions);

            int j = Ints.checkedCast(getFlatIndex(resultShape, resultStride, shapeIndices));

            newBuffer.plus(j, buffer.get(i));
        }

        return create(newBuffer, resultShape, resultStride);
    }

    @Override
    public TENSOR cumSumInPlace(int requestedDimension) {

        int dimension = requestedDimension >= 0 ? requestedDimension : requestedDimension + shape.length;
        TensorShapeValidation.checkDimensionExistsInShape(dimension, shape);
        long[] index = new long[shape.length];
        int[] dimensionOrder = ArrayUtils.remove(dimensionRange(0, shape.length), dimension);

        do {

            B sum = getFactory().createNew(1);
            for (int i = 0; i < shape[dimension]; i++) {

                index[dimension] = i;

                int j = Ints.checkedCast(getFlatIndex(shape, stride, index));
                buffer.plus(j, sum.get(0));
                sum.set(buffer.get(j), 0);
            }

        } while (incrementIndexByShape(shape, index, dimensionOrder));

        return set(buffer, shape, stride);
    }

}
