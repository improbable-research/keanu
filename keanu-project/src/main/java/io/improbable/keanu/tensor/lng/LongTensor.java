package io.improbable.keanu.tensor.lng;

import io.improbable.keanu.tensor.FixedPointTensor;
import io.improbable.keanu.tensor.TensorFactories;
import org.apache.commons.lang3.ArrayUtils;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;

public interface LongTensor extends FixedPointTensor<Long, LongTensor> {

    static LongTensor create(long value, long[] shape) {
        return TensorFactories.longTensorFactory.create(value, shape);
    }

    static LongTensor create(long[] values, long... shape) {
        return TensorFactories.longTensorFactory.create(values, shape);
    }

    static LongTensor create(long... values) {
        return create(values, values.length);
    }

    static LongTensor ones(long... shape) {
        return TensorFactories.longTensorFactory.ones(shape);
    }

    static LongTensor eye(int n) {
        return TensorFactories.longTensorFactory.eye(n);
    }

    static LongTensor zeros(long... shape) {
        return TensorFactories.longTensorFactory.zeros(shape);
    }

    static LongTensor scalar(long scalarValue) {
        return TensorFactories.longTensorFactory.scalar(scalarValue);
    }

    static LongTensor vector(long... values) {
        return create(values, values.length);
    }

    static LongTensor stack(int dimension, LongTensor... toStack) {
        long[] shape = toStack[0].getShape();
        int absoluteDimension = getAbsoluteDimension(dimension, shape.length + 1);
        long[] stackedShape = ArrayUtils.insert(absoluteDimension, shape, 1);

        LongTensor[] reshaped = new LongTensor[toStack.length];
        for (int i = 0; i < toStack.length; i++) {
            reshaped[i] = toStack[i].reshape(stackedShape);
        }

        return concat(absoluteDimension, reshaped);
    }

    static LongTensor concat(LongTensor... toConcat) {
        return concat(0, toConcat);
    }

    static LongTensor concat(int dimension, LongTensor... toConcat) {
        return TensorFactories.longTensorFactory.concat(dimension, toConcat);
    }

    static LongTensor min(LongTensor a, LongTensor b) {
        return a.duplicate().minInPlace(b);
    }

    static LongTensor max(LongTensor a, LongTensor b) {
        return a.duplicate().maxInPlace(b);
    }

}
