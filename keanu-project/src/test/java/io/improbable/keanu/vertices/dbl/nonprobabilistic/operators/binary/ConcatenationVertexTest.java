package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Assert;
import org.junit.Test;

public class ConcatenationVertexTest {

    @Test
    public void canConcatVectorsOfSameSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        UniformVertex a1 = new UniformVertex(0.0, 1.0);
        a1.setValue(new double[]{4, 5, 6});

        UniformVertex a2 = new UniformVertex(0.0, 1.0);
        a2.setValue(new double[]{7, 8, 9});

        ConcatenationVertex concatAlongZero = new ConcatenationVertex(0, a, a1);
        ConcatenationVertex concatAlongOne = new ConcatenationVertex(1, a, a1, a2);

        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, concatAlongZero.getValue().asFlatDoubleArray(), 0.001);
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, concatAlongOne.getValue().asFlatDoubleArray(), 0.001);

        Assert.assertArrayEquals(new int[]{2, 3}, concatAlongZero.getShape());
        Assert.assertArrayEquals(new int[]{1, 9}, concatAlongOne.getShape());
    }

    @Test
    public void canConcatVectorsOfDifferentSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        UniformVertex a1 = new UniformVertex(0.0, 1.0);
        a1.setValue(new double[]{4, 5, 6, 7, 8, 9});

        ConcatenationVertex concatAlongZero = new ConcatenationVertex(1, a, a1);

        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, concatAlongZero.getValue().asFlatDoubleArray(), 0.001);
        Assert.assertArrayEquals(new int[]{1, 9}, concatAlongZero.getShape());
    }

    @Test (expected = IllegalArgumentException.class)
    public void errorThrownOnConcatOfWrongSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        UniformVertex a1 = new UniformVertex(0.0, 1.0);
        a1.setValue(new double[]{4, 5, 6, 7, 8, 9});

        ConcatenationVertex concatAlongZero = new ConcatenationVertex(0, a, a1);

        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, concatAlongZero.getValue().asFlatDoubleArray(), 0.001);
        Assert.assertArrayEquals(new int[]{1, 9}, concatAlongZero.getShape());
    }

    @Test
    public void canConcatMatricesOfSameSize() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        ConcatenationVertex concatZero = new ConcatenationVertex(0, m, a);

        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 10, 15, 20, 25}, concatZero.getValue().asFlatDoubleArray(), 0.001);
        Assert.assertArrayEquals(new int[]{4, 2}, concatZero.getShape());

        ConcatenationVertex concatOne = new ConcatenationVertex(1, m, a);

        Assert.assertArrayEquals(new double[]{1, 2, 10, 15, 3, 4, 20, 25}, concatOne.getValue().asFlatDoubleArray(), 0.001);
        Assert.assertArrayEquals(new int[]{2, 4}, concatOne.getShape());

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
        Assert.assertArrayEquals(
            c.getDualNumber().getPartialDerivatives().withRespectTo(a).asFlatDoubleArray(),
            concatPartial.withRespectTo(a).asFlatDoubleArray(),
            0.0001
        );
        Assert.assertArrayEquals(
            d.getDualNumber().getPartialDerivatives().withRespectTo(b).asFlatDoubleArray(),
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
        Assert.assertArrayEquals(
            c.getDualNumber().getPartialDerivatives().withRespectTo(a).asFlatDoubleArray(),
            concatPartial.withRespectTo(a).asFlatDoubleArray(),
            0.0001
        );
        Assert.assertArrayEquals(
            d.getDualNumber().getPartialDerivatives().withRespectTo(b).asFlatDoubleArray(),
            concatPartial.withRespectTo(b).asFlatDoubleArray(),
            0.0001
        );
    }
}
