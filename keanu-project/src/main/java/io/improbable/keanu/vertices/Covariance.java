package io.improbable.keanu.vertices;

import java.util.Collections;
import java.util.List;

import org.nd4j.linalg.api.blas.BlasException;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.ScalarDoubleTensor;

public class Covariance {
    private final DoubleTensor matrix;
    private final List<VertexId> vertexIds;

    public Covariance(DoubleTensor matrix, VertexId... vertexIds) {
        int[] shape = matrix.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("The covariance matrix must be rank 2");
        } else if (shape[0] != shape[1]) {
            throw new IllegalArgumentException("The covariance matrix must be square");
        } else if (vertexIds.length != shape[0]) {
            throw new IllegalArgumentException("You must pass in " + shape[0] + " VertexIds, to match the dimension of the matrix");
        } else if (!isSymmetric(matrix)) {
            throw new IllegalArgumentException("The covariance matrix must be symmetric");
        }
        try {
            matrix.choleskyDecomposition();
        } catch (BlasException e) {
            throw new IllegalArgumentException("The covariance matrix must be positive semi-definite", e);
        }
        this.matrix = matrix.duplicate();
        this.vertexIds = ImmutableList.copyOf(vertexIds);
    }

    private boolean isSymmetric(DoubleTensor matrix) {
        int dimension = matrix.getShape()[0];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (!matrix.getValue(i, j).equals( matrix.getValue(j, i))) {
                    return false;
                }
            }
        }
        return true;
    }

    public DoubleTensor asTensor() {
        return matrix.duplicate();
    }

    public DoubleTensor getSubMatrix(VertexId... subset) {
        int[] indices = getIndicesOf(subset);
        DoubleTensor subMatrix = getSubMatrix(matrix, indices, 0);
        subMatrix = getSubMatrix(subMatrix, indices, 1);
        if (subMatrix.isScalar()) {
            return new ScalarDoubleTensor(subMatrix.getValue(0));
        } else {
            return subMatrix;
        }
    }

    private DoubleTensor getSubMatrix(DoubleTensor original, int[] indices, int dimension) {
        ImmutableList.Builder<DoubleTensor> slices = ImmutableList.builder();
        for (int index : indices) {
            slices.add(original.slice(dimension, index));
        }

        return DoubleTensor.concat(dimension, slices.build().toArray(new DoubleTensor[0]));
    }

    private int[] getIndicesOf(VertexId[] subset) {
        ImmutableList.Builder<Integer> indices = ImmutableList.builder();

        for (VertexId vertexId : subset) {
            int i = Collections.indexOfSubList(vertexIds, ImmutableList.of(vertexId));
            if (i == -1) {
                throw new IllegalArgumentException("Cannot find VertexId " + vertexId);
            }
            indices.add(i);
        }

        return indices.build().stream().mapToInt(Integer::intValue).toArray();
    }
}
