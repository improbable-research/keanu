package io.improbable.keanu.tensor.jvm;

import com.google.common.base.Preconditions;
import io.improbable.keanu.tensor.TensorShape;
import lombok.Value;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;

import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getShapeIndices;

@Value
public class DimensionIndexMapper implements IndexMapper {

    final int dimension;
    final long index;
    final long[] sourceShape;
    final long[] sourceStride;
    final long[] resultShape;
    final long[] resultStride;

    public DimensionIndexMapper(long[] sourceShape, long[] sourceStride, int dimension, long index) {
        Preconditions.checkArgument(dimension < sourceShape.length, "Invalid dimension " + dimension + " for shape " + Arrays.toString(sourceShape));
        Preconditions.checkArgument(index < sourceShape[dimension], "Invalid index at dimension " + dimension + " at index " + index + " for shape " + Arrays.toString(sourceShape));
        this.dimension = dimension;
        this.index = index;
        this.sourceShape = sourceShape;
        this.sourceStride = sourceStride;
        this.resultShape = ArrayUtils.remove(sourceShape, dimension);
        this.resultStride = TensorShape.getRowFirstStride(resultShape);
    }

    @Override
    public long getSourceIndexFromResultIndex(long resultIndex) {
        long[] shapeIndices = ArrayUtils.insert(dimension, getShapeIndices(resultShape, resultStride, resultIndex), index);
        return getFlatIndex(sourceShape, sourceStride, shapeIndices);
    }
}
