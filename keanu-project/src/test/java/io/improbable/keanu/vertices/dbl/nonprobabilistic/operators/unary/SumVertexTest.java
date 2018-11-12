package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class SumVertexTest {

    @Test
    public void doesSumAllDimensions() {
        DoubleVertex a = new UniformVertex(new long[]{1, 5}, 0, 10);
        a.setValue(new double[]{1, 2, 3, 4, 5});

        DoubleVertex summed = a.sum();

        assertEquals(1 + 2 + 3 + 4 + 5, summed.eval().scalar(), 1e-5);
    }

    @Test
    public void doesSumAllSpecifiedDimensions() {
        DoubleVertex a = new UniformVertex(new long[]{1, 5}, 0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4, 5}, 1, 5));

        DoubleVertex summed = a.sum(0, 1);
        DoubleTensor expected = DoubleTensor.scalar(1 + 2 + 3 + 4 + 5);

        assertEquals(expected, summed.eval());
    }

    @Test
    public void doesSumSpecifiedDimensions() {
        long[] shape = {2, 2, 2, 2};
        DoubleVertex a = new UniformVertex(shape, 0, 10);
        a.setValue(DoubleTensor.arange(0, TensorShape.getLength(shape)).reshape(shape));

        DoubleVertex summed = a.sum(0, 2);

        DoubleTensor expected = DoubleTensor.create(new double[]{20, 24, 36, 40}, 2, 2);

        assertEquals(expected, summed.eval());
    }

    @Test
    public void doesCalculateCorrectShape() {
        long[] shape = {2, 3, 4, 5, 6, 1};
        DoubleVertex a = new UniformVertex(shape, 0, 10);
        DoubleTensor highrank = DoubleTensor.arange(0, TensorShape.getLength(shape)).reshape(shape);
        a.setValue(highrank);

        assertArrayEquals(new long[]{3, 5, 6, 1}, a.sum(0, 2).getShape());
        assertArrayEquals(new long[]{5, 6}, a.sum(0, 1, 2, 5).getShape());
        assertArrayEquals(new long[]{6, 1}, a.sum(0, 1, 2, 3).getShape());
        assertArrayEquals(new long[]{2}, a.sum(1, 2, 3, 4, 5).getShape());
        assertArrayEquals(new long[]{3}, a.sum(0, 2, 3, 4, 5).getShape());
        assertArrayEquals(new long[]{}, a.sum(0, 1, 2, 3, 4, 5).getShape());
    }

    @Test
    public void doesSumScalarCorrectly() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(2);

        assertArrayEquals(new long[0], a.sum().getShape());
    }

    @Test
    public void doesSumAllSimpleAutoDiff() {
        DoubleVertex a = new UniformVertex(new long[]{2, 2, 2}, 0, 10);
        a.setValue(a.sample());

        SumVertex b = a.sum();

        DoubleTensor dbdaForward = b.getDerivativeWrtLatents().withRespectTo(a);
        DoubleTensor dbdaReverse = Differentiator.reverseModeAutoDiff(b, a).withRespectTo(a);

        DoubleTensor expectedDbDa = DoubleTensor.ones(new long[]{2, 2, 2});

        assertThat(dbdaForward, equalTo(expectedDbDa));
        assertThat(dbdaReverse, equalTo(expectedDbDa));
    }

    @Test
    public void doesSumSpecifiedSimpleAutoDiff() {
        DoubleVertex a = new UniformVertex(new long[]{2, 2}, 0, 10);
        a.setValue(DoubleTensor.arange(0, 4).reshape(2, 2));

        int sumDimension = 1;
        SumVertex b = a.sum(sumDimension);

        DoubleTensor dbdaForward = b.getDerivativeWrtLatents().withRespectTo(a);
        DoubleTensor dbdaReverse = Differentiator.reverseModeAutoDiff(b, a).withRespectTo(a);

        DoubleTensor expectedDbDa = DoubleTensor.eye(4).reshape(2, 2, 2, 2).sum(sumDimension).reshape(2, 2, 2);

        assertThat(dbdaForward, equalTo(expectedDbDa));
        assertThat(dbdaReverse, equalTo(expectedDbDa));
    }

    @Test
    public void doesSumSpecifiedRank3AutoDiff() {
        DoubleVertex a = new UniformVertex(new long[]{2, 2, 2}, 0, 10);
        a.setValue(DoubleTensor.arange(0, 8).reshape(2, 2, 2));

        int sumDimension = 1;
        SumVertex b = a.sum(sumDimension);

        DoubleTensor dbdaForward = b.getDerivativeWrtLatents().withRespectTo(a);
        DoubleTensor dbdaReverse = Differentiator.reverseModeAutoDiff(b, a).withRespectTo(a);

        DoubleTensor expectedDbDa = DoubleTensor.eye(8).reshape(2, 2, 2, 2, 2, 2).sum(sumDimension).reshape(2, 2, 2, 2, 2);

        assertThat(dbdaForward, equalTo(expectedDbDa));
        assertThat(dbdaReverse, equalTo(expectedDbDa));
    }

    @Test
    public void canDoSumAutoDiffWhenSumIsNotWrtOrOf() {
        DoubleVertex a = new UniformVertex(new long[]{2, 3}, 0, 10);
        a.setValue(DoubleTensor.arange(0, 6).reshape(2, 3));

        DoubleVertex d = a.sum();

        DoubleVertex e = new UniformVertex(new long[]{2, 2}, 0, 10);
        e.setValue(DoubleTensor.arange(4, 8).reshape(2, 2));

        MultiplicationVertex f = d.times(e);

        DoubleTensor dfdaForward = f.getDerivativeWrtLatents().withRespectTo(a);

        PartialDerivatives dfdx = Differentiator.reverseModeAutoDiff(f, a);
        DoubleTensor dfdaReverse = dfdx.withRespectTo(a);

        DoubleTensor expectedDfdx = DoubleTensor.create(new double[]{
            4, 4, 4,
            4, 4, 4,
            5, 5, 5,
            5, 5, 5,
            6, 6, 6,
            6, 6, 6,
            7, 7, 7,
            7, 7, 7
        }, 2, 2, 2, 3);

        assertThat(dfdaReverse, equalTo(expectedDfdx));
        assertThat(dfdaForward, equalTo(expectedDfdx));
    }

    @Test
    public void canDoSumAutoDiffWhenOfIsScalar() {
        DoubleVertex a = new UniformVertex(new long[]{2, 3}, 0, 10);
        a.setValue(DoubleTensor.arange(0, 6).reshape(2, 3));

        DoubleVertex d = a.sum();

        DoubleVertex e = new UniformVertex(0, 10);
        e.setValue(2);

        MultiplicationVertex f = d.times(e);

        DoubleTensor dfdaForward = f.getDerivativeWrtLatents().withRespectTo(a);

        PartialDerivatives dfdx = Differentiator.reverseModeAutoDiff(f, a);
        DoubleTensor dfdaReverse = dfdx.withRespectTo(a);

        DoubleTensor expectedDfdx = DoubleTensor.create(new double[]{
            2, 2, 2,
            2, 2, 2
        }, 2, 3);

        assertThat(dfdaForward, equalTo(expectedDfdx));
        assertThat(dfdaReverse, equalTo(expectedDfdx));
    }

    @Test
    public void canDoSumAutoDiffWhenWrtIsScalar() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(2);

        DoubleVertex d = a.sum();

        DoubleVertex e = new UniformVertex(0, 10);
        e.setValue(new double[]{1, 2, 3});

        MultiplicationVertex f = d.times(e);

        DoubleTensor dfdaForward = f.getDerivativeWrtLatents().withRespectTo(a);

        PartialDerivatives dfdx = Differentiator.reverseModeAutoDiff(f, a);
        DoubleTensor dfdaReverse = dfdx.withRespectTo(a);

        DoubleTensor expectedDfda = DoubleTensor.create(1, 2, 3);

        assertThat(dfdaForward, equalTo(expectedDfda));
        assertThat(dfdaReverse, equalTo(expectedDfda));
    }

    @Test
    public void changesMatchGradientWhenSummingAll() {
        DoubleVertex inputVertex = new UniformVertex(new long[]{2, 2, 2}, -10.0, 10.0);
        inputVertex.setValue(DoubleTensor.arange(0, 8).reshape(2, 2, 2));
        MultiplicationVertex outputVertex = inputVertex.sum().times(inputVertex);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 1e-6, 1e-10);
    }

    @Test
    public void changesMatchGradientWhenSummingSpecificDimensions() {
        DoubleVertex inputVertex = new UniformVertex(new long[]{2, 2, 2}, -10.0, 10.0);
        inputVertex.setValue(DoubleTensor.arange(0, 8).reshape(2, 2, 2));

        MultiplicationVertex outputVertex = inputVertex.sum(0)
            .times(
                inputVertex.sum(1)
            ).times(
                inputVertex.sum(2)
            ).times(
                inputVertex.sum()
            );

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 1e-6, 1e-6);
    }

}
