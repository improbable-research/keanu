package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class ConcatenationVertexTest {

    @Test
    public void canConcatVectorsOfSameSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        UniformVertex b = new UniformVertex(0.0, 1.0);
        b.setValue(new double[]{4, 5, 6});

        UniformVertex c = new UniformVertex(0.0, 1.0);
        c.setValue(new double[]{7, 8, 9});

        ConcatenationVertex concatZero = new ConcatenationVertex(0, a, b);
        ConcatenationVertex concatOne = new ConcatenationVertex(1, a, b, c);

        Assert.assertArrayEquals(new int[]{2, 3}, concatZero.getShape());
        Assert.assertArrayEquals(new int[]{1, 9}, concatOne.getShape());

        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, concatZero.getValue().asFlatDoubleArray(), 0.001);
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, concatOne.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatVectorsOfDifferentSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        UniformVertex b = new UniformVertex(0.0, 1.0);
        b.setValue(new double[]{4, 5, 6, 7, 8, 9});

        ConcatenationVertex concatZero = new ConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[]{1, 9}, concatZero.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, concatZero.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatScalarToVector() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        DoubleVertex b = new ConstantDoubleVertex(4.0);

        ConcatenationVertex concat = new ConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[]{1, 4}, concat.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4}, concat.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatVectorToScalar() {
        DoubleVertex a = new ConstantDoubleVertex(1.0);

        UniformVertex b = new UniformVertex(0.0, 1.0);
        b.setValue(new double[]{2, 3, 4});

        ConcatenationVertex concat = new ConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[]{1, 4}, concat.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4}, concat.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test(expected = IllegalArgumentException.class)
    public void errorThrownOnConcatOfWrongSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        UniformVertex a1 = new UniformVertex(0.0, 1.0);
        a1.setValue(new double[]{4, 5, 6, 7, 8, 9});

        new ConcatenationVertex(0, a, a1);
    }

    @Test
    public void canConcatMatricesOfSameSize() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        ConcatenationVertex concatZero = new ConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(new int[]{4, 2}, concatZero.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 10, 15, 20, 25}, concatZero.getValue().asFlatDoubleArray(), 0.001);

        ConcatenationVertex concatOne = new ConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[]{2, 4}, concatOne.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 10, 15, 3, 4, 20, 25}, concatOne.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatHighDimensionalShapes() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 20, 30, 40, 50, 60, 70, 80}, 2, 2, 2));

        ConcatenationVertex concatZero = new ConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(
            new double[]{1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80},
            concatZero.getValue().asFlatDoubleArray(),
            0.001
        );
        Assert.assertArrayEquals(new int[]{4, 2, 2}, concatZero.getShape());

        ConcatenationVertex concatThree = new ConcatenationVertex(2, a, b);
        Assert.assertArrayEquals(
            new double[]{1, 2, 10, 20, 3, 4, 30, 40, 5, 6, 50, 60, 7, 8, 70, 80},
            concatThree.getValue().asFlatDoubleArray(),
            0.001
        );
        Assert.assertArrayEquals(new int[]{2, 2, 4}, concatThree.getShape());
    }

    @Test
    public void canConcatenateSimpleAutoDiff() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex c = a.times(b);
        DoubleVertex d = a.plus(b);

        ConcatenationVertex concat = new ConcatenationVertex(0, c, d);
        PartialDerivatives concatPartial = concat.getDualNumber().getPartialDerivatives();

        PartialDerivatives cPartial = c.getDualNumber().getPartialDerivatives();
        PartialDerivatives dPartial = d.getDualNumber().getPartialDerivatives();

        Assert.assertArrayEquals(
            cPartial.withRespectTo(a).concat(0, dPartial.withRespectTo(a)).asFlatDoubleArray(),
            concatPartial.withRespectTo(a).asFlatDoubleArray(),
            0.0001
        );
        Assert.assertArrayEquals(
            cPartial.withRespectTo(b).concat(0, dPartial.withRespectTo(b)).asFlatDoubleArray(),
            concatPartial.withRespectTo(b).asFlatDoubleArray(),
            0.0001
        );
    }

    @Test
    public void canConcatenateSimpleAutoDiffForwardNoSharedParentsDimensionOne() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex c = new UniformVertex(0, 10);
        c.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex d = new UniformVertex(0, 10);
        d.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex e = a.times(b);
        DoubleVertex f = c.plus(d);

        ConcatenationVertex concat = new ConcatenationVertex(1, e, f);

        PartialDerivatives forward = Differentiator.forwardModeAutoDiff(concat, Arrays.asList(a, b, c, d));
        //TODO: make this work for backward

        Assert.assertArrayEquals(new int[]{2, 4, 2, 2}, forward.withRespectTo(a).getShape());
        Assert.assertArrayEquals(new int[]{2, 4, 2, 2}, forward.withRespectTo(b).getShape());
        Assert.assertArrayEquals(new int[]{2, 4, 2, 2}, forward.withRespectTo(c).getShape());
        Assert.assertArrayEquals(new int[]{2, 4, 2, 2}, forward.withRespectTo(d).getShape());
    }

    @Test
    public void canConcatenateSimpleAutoDiffForwardSharedParents() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5}, 1, 1));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex d = new UniformVertex(0, 10);
        d.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex e = a.times(b);
        DoubleVertex f = b.plus(d);

        ConcatenationVertex concat = new ConcatenationVertex(0, e, f);

        PartialDerivatives forward = Differentiator.forwardModeAutoDiff(concat, Arrays.asList(a, b, d));

        Assert.assertArrayEquals(new int[]{4, 2, 1, 1}, forward.withRespectTo(a).getShape());
        Assert.assertArrayEquals(new int[]{4, 2, 2, 2}, forward.withRespectTo(b).getShape());
        Assert.assertArrayEquals(new int[]{4, 2, 2, 2}, forward.withRespectTo(d).getShape());
    }

    @Test
    public void canCalculateValueOfConcatenated() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex c = a.times(b);
        DoubleVertex d = a.plus(b);

        ConcatenationVertex concat = new ConcatenationVertex(0, c, d);
        DoubleTensor dualNumberValue = concat.getDualNumber().getValue();

        Assert.assertArrayEquals(
            new double[]{50, 90, 140, 200, 15, 21, 27, 33},
            dualNumberValue.asFlatDoubleArray(),
            0.0001
        );
    }


    @Test
    public void canConcatenateAutoDiffMatricesAlongDimensionZero() {
        DoubleVertex sharedMatrix = new UniformVertex(0, 10);
        sharedMatrix.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex c = sharedMatrix.matrixMultiply(a);
        DoubleVertex d = sharedMatrix.matrixMultiply(b);

        DoubleTensor dCdshared = c.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);
        DoubleTensor dDdshared = d.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);

        ConcatenationVertex concat = new ConcatenationVertex(0, c, d);
        PartialDerivatives concatPartial = concat.getDualNumber().getPartialDerivatives();

        Assert.assertArrayEquals(
            dCdshared.concat(0, dDdshared).asFlatDoubleArray(),
            concatPartial.withRespectTo(sharedMatrix).asFlatDoubleArray(),
            0.0001
        );

        DoubleTensor cwrtA = c.getDualNumber().getPartialDerivatives().withRespectTo(a);
        Assert.assertArrayEquals(
            cwrtA.concat(0, DoubleTensor.zeros(cwrtA.getShape())).asFlatDoubleArray(),
            concatPartial.withRespectTo(a).asFlatDoubleArray(),
            0.0001
        );

        DoubleTensor dwrtB = d.getDualNumber().getPartialDerivatives().withRespectTo(b);
        Assert.assertArrayEquals(
            DoubleTensor.zeros(dwrtB.getShape()).concat(0, dwrtB).asFlatDoubleArray(),
            concatPartial.withRespectTo(b).asFlatDoubleArray(),
            0.0001
        );
    }

    @Test
    public void canConcatenateAutoDiffMatricesAlongDimensionOne() {
        DoubleVertex sharedMatrix = new UniformVertex(0, 10);
        sharedMatrix.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex c = sharedMatrix.matrixMultiply(a);
        DoubleVertex d = sharedMatrix.matrixMultiply(b);

        DoubleTensor dCdshared = c.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);
        DoubleTensor dDdshared = d.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);

        ConcatenationVertex concat = new ConcatenationVertex(1, c, d);
        PartialDerivatives concatPartial = concat.getDualNumber().getPartialDerivatives();

        Assert.assertArrayEquals(
            dCdshared.concat(1, dDdshared).asFlatDoubleArray(),
            concatPartial.withRespectTo(sharedMatrix).asFlatDoubleArray(),
            0.0001
        );

        DoubleTensor cwrtA = c.getDualNumber().getPartialDerivatives().withRespectTo(a);
        Assert.assertArrayEquals(
            cwrtA.concat(1, DoubleTensor.zeros(cwrtA.getShape())).asFlatDoubleArray(),
            concatPartial.withRespectTo(a).asFlatDoubleArray(),
            0.0001
        );

        DoubleTensor dwrtB = d.getDualNumber().getPartialDerivatives().withRespectTo(b);
        Assert.assertArrayEquals(
            DoubleTensor.zeros(dwrtB.getShape()).concat(1, dwrtB).asFlatDoubleArray(),
            concatPartial.withRespectTo(b).asFlatDoubleArray(),
            0.0001
        );
    }

    @Test
    public void canConcatenateAutoDiffOfManyMatricesAlongDimensionZero() {
        DoubleVertex sharedMatrix = new UniformVertex(0, 10);
        sharedMatrix.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex f = new UniformVertex(0, 10);
        f.setValue(DoubleTensor.create(new double[]{90, 91, 92, 93}, 2, 2));

        DoubleVertex c = sharedMatrix.matrixMultiply(a);
        DoubleVertex d = sharedMatrix.matrixMultiply(b);
        DoubleVertex e = sharedMatrix.matrixMultiply(f);

        DoubleTensor dCdshared = c.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);
        DoubleTensor dDdshared = d.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);
        DoubleTensor dEdshared = e.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);

        ConcatenationVertex concat = new ConcatenationVertex(0, c, d, e);
        PartialDerivatives concatPartial = concat.getDualNumber().getPartialDerivatives();

        Assert.assertArrayEquals(
            dCdshared.concat(0, dDdshared, dEdshared).asFlatDoubleArray(),
            concatPartial.withRespectTo(sharedMatrix).asFlatDoubleArray(),
            0.0001
        );

        DoubleTensor cwrtA = c.getDualNumber().getPartialDerivatives().withRespectTo(a);
        Assert.assertArrayEquals(
            cwrtA.concat(0, DoubleTensor.zeros(cwrtA.getShape()), DoubleTensor.zeros(cwrtA.getShape())).asFlatDoubleArray(),
            concatPartial.withRespectTo(a).asFlatDoubleArray(),
            0.0001
        );

        DoubleTensor dwrtB = d.getDualNumber().getPartialDerivatives().withRespectTo(b);
        Assert.assertArrayEquals(
            DoubleTensor.zeros(dwrtB.getShape()).concat(0, dwrtB, DoubleTensor.zeros(dwrtB.getShape())).asFlatDoubleArray(),
            concatPartial.withRespectTo(b).asFlatDoubleArray(),
            0.0001
        );

        DoubleTensor ewrtC = e.getDualNumber().getPartialDerivatives().withRespectTo(f);
        Assert.assertArrayEquals(
            DoubleTensor.zeros(ewrtC.getShape()).concat(0, DoubleTensor.zeros(ewrtC.getShape()), ewrtC).asFlatDoubleArray(),
            concatPartial.withRespectTo(f).asFlatDoubleArray(),
            0.0001
        );
    }

    @Test
    public void canSplit() {

        int dim = 2;
        DoubleTensor A = DoubleTensor.arange(0, 24).reshape(2, 3, 1, 4);
        DoubleTensor B = DoubleTensor.arange(24, 96).reshape(2, 3, 3, 4);
        DoubleTensor C = DoubleTensor.arange(96, 144).reshape(2, 3, 2, 4);

        DoubleTensor D = A.concat(dim, B, C);
        List<DoubleTensor> splitTensor = split(D, dim, new int[]{1, 4, 6});

        DoubleTensor[] concatList = new DoubleTensor[]{A, B, C};
        for (int i = 0; i < splitTensor.size(); i++) {
            assertEquals(concatList[i], splitTensor.get(i));
        }

    }

    @Test
    public void canSplitHighRank() {
        assertCanSplit(new int[]{2, 3, 4, 5, 7, 2}, new int[]{3, 2, 6}, 1);
    }

    @Test
    public void canSplitEndDimension() {
        assertCanSplit(new int[]{2, 3, 4, 5}, new int[]{3, 4, 2}, 3);
    }

    @Test
    public void canSplitFirstDimension() {
        assertCanSplit(new int[]{2, 3, 4, 5, 7, 2}, new int[]{3, 4, 2, 6, 9, 2}, 0);
    }

    private void assertCanSplit(int[] baseShape, int[] concatenatedIndices, int concatenatedDimension) {

        int[] splitIndices = new int[concatenatedIndices.length];
        List<DoubleTensor> toConcat = new ArrayList<>();

        long previousEndLength = 0;
        int splitPosition = 0;
        for (int i = 0; i < concatenatedIndices.length; i++) {
            int[] shape = Arrays.copyOf(baseShape, baseShape.length);
            shape[concatenatedDimension] = concatenatedIndices[i];

            splitIndices[i] = splitPosition + concatenatedIndices[i];
            splitPosition = splitIndices[i];

            long newEndLength = previousEndLength + TensorShape.getLength(shape);
            toConcat.add(DoubleTensor.arange(previousEndLength, newEndLength).reshape(shape));
            previousEndLength = newEndLength;
        }

        DoubleTensor D = toConcat.get(0).concat(concatenatedDimension, toConcat.subList(1, toConcat.size()).toArray(new DoubleTensor[toConcat.size() - 1]));
        List<DoubleTensor> splitTensor = split(D, concatenatedDimension, splitIndices);

        for (int i = 0; i < splitTensor.size(); i++) {
            assertEquals(toConcat.get(i), splitTensor.get(i));
        }
    }

    private List<DoubleTensor> split(DoubleTensor tensor, int dimension, int[] splitAtIndices) {

        int[] tensorShape = tensor.getShape();
        int[] dimensionRange = TensorShape.dimensionRange(0, tensorShape.length);
        int[] moveDimToZero = TensorShape.moveAxis(dimension, 0, dimensionRange);
        int[] moveZeroToDim = TensorShape.moveAxis(0, dimension, dimensionRange);

        DoubleTensor permutedTensor = tensor.permute(moveDimToZero).reshape(1, (int) tensor.getLength());

        double[] rawBuffer = permutedTensor.asFlatDoubleArray();

        List<DoubleTensor> splitTensor = new ArrayList<>();

        int previousIndex = 0;
        int rawBufferPosition = 0;
        for (int index : splitAtIndices) {

            int[] subTensorShape = Arrays.copyOf(tensorShape, tensorShape.length);
            int subTensorLengthInDimension = index - previousIndex;

            if (subTensorLengthInDimension > tensorShape[dimension] || subTensorLengthInDimension < 0) {
                throw new IllegalArgumentException("Invalid index to split on " + index + " at " + dimension + " for tensor of shape " + Arrays.toString(tensorShape));
            }

            subTensorShape[dimension] = subTensorLengthInDimension;
            previousIndex = index;

            int subTensorLength = (int) TensorShape.getLength(subTensorShape);

            double[] buffer = new double[subTensorLength];
            System.arraycopy(rawBuffer, rawBufferPosition, buffer, 0, buffer.length);

            int[] subTensorPermutedShape = TensorShape.moveAxis(dimension, 0, subTensorShape);

            DoubleTensor subTensor = DoubleTensor.create(buffer, new int[]{1, subTensorLength}).reshape(subTensorPermutedShape).permute(moveZeroToDim);

            splitTensor.add(subTensor);

            rawBufferPosition += buffer.length;

        }

        return splitTensor;
    }

}
