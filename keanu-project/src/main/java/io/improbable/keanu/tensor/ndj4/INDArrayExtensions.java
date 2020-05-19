package io.improbable.keanu.tensor.ndj4;

import com.google.common.primitives.Ints;
import org.bytedeco.javacpp.indexer.BooleanIndexer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;

public class INDArrayExtensions {

    public static boolean[] asBoolean(INDArray array) {
        if (array.dataType() != DataType.BOOL) {
            array = array.castTo(DataType.BOOL);
        }

        boolean[] buffer = new boolean[Ints.checkedCast(array.length())];

        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = ((BooleanIndexer) (array.data()).indexer()).get(i);
        }

        return buffer;
    }

    public static List<INDArray> split(INDArray tensor, int dimension, long... splitAtIndices) {
        long[] shape = tensor.shape();
        dimension = getAbsoluteDimension(dimension, tensor.rank());

        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Invalid dimension to split on " + dimension);
        }

        Nd4j.getCompressor().autoDecompress(tensor);

        List<INDArray> splits = new ArrayList<>();
        long previousSplitIndex = 0;
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

            splits.add(tensor.get(indices));
        }

        return splits;
    }

    /**
     * Cumulative prod along a dimension. This code is copied from the
     * cumSumi implementation in org.nd4j.linalg.api.ndarray.BaseNDArray.java
     *
     * @param array     array to cumProd
     * @param dimension the dimension to perform cumulative product along
     * @return the cumulative product along the specified dimension
     */
    public static INDArray cumProd(INDArray array, int dimension) {

        if (array.isScalar() || array.isEmpty())
            return array;

        if (array.isVector()) {
            double s = 1.0;
            for (int i = 0; i < array.length(); i++) {
                s *= array.getDouble(i);
                array.putScalar(i, s);
            }
        } else if (dimension == Integer.MAX_VALUE) {
            INDArray flattened = array.ravel();
            double prevVal = flattened.getDouble(0);
            for (int i = 1; i < flattened.length(); i++) {
                double d = prevVal * flattened.getDouble(i);
                flattened.putScalar(i, d);
                prevVal = d;
            }

            return flattened;
        } else {
            for (int i = 0; i < array.vectorsAlongDimension(dimension); i++) {
                INDArray vec = array.vectorAlongDimension(i, dimension);
                cumProd(vec, 0);
            }
        }


        return array;
    }
}
