package io.improbable.keanu.tensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;

public class INDArrayExtensions {

    public static INDArray castToInteger(INDArray tensor, boolean duplicate) {
        INDArray tensorToDropFractionOn = duplicate ? tensor.dup() : tensor;
        INDArray sign = Transforms.sign(tensorToDropFractionOn);
        Transforms.floor(Transforms.abs(tensorToDropFractionOn, false), false).muli(sign);
        return tensorToDropFractionOn;
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
}
