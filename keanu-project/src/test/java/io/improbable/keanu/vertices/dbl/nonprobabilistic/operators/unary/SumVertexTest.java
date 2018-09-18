package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesGradient;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class SumVertexTest {

    @Test
    public void doesSum() {
        DoubleVertex a = new UniformVertex(new int[]{1, 5}, 0, 10);
        a.setValue(new double[]{1, 2, 3, 4, 5});

        DoubleVertex summed = a.sum();

        assertEquals(1 + 2 + 3 + 4 + 5, summed.eval().scalar(), 1e-5);
    }

    @Test
    public void doesSumSimpleAutoDiff() {
        DoubleVertex a = new UniformVertex(new int[]{2, 2, 2}, 0, 10);
        a.setValue(a.sample());

        DoubleVertex b = a.sum();

        DoubleTensor dbdaForward = b.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dbdaReverse = Differentiator.reverseModeAutoDiff(b, a).withRespectTo(a);

        DoubleTensor expectedDbDa = DoubleTensor.ones(new int[]{1, 1, 2, 2, 2});

        assertThat(dbdaForward, equalTo(expectedDbDa));
        assertThat(dbdaReverse, equalTo(expectedDbDa));
    }

    @Test
    public void canDoSumAutoDiffWhenSumIsNotWrtOrOf() {
        DoubleVertex a = new UniformVertex(new int[]{2, 3}, 0, 10);
        a.setValue(DoubleTensor.arange(0, 6).reshape(2, 3));

        DoubleVertex d = a.sum();

        DoubleVertex e = new UniformVertex(new int[]{2, 2}, 0, 10);
        e.setValue(DoubleTensor.arange(4, 8).reshape(2, 2));

        DoubleVertex f = d.times(e);

        DoubleTensor dfdaForward = f.getDualNumber().getPartialDerivatives().withRespectTo(a);

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
        DoubleVertex a = new UniformVertex(new int[]{2, 3}, 0, 10);
        a.setValue(DoubleTensor.arange(0, 6).reshape(2, 3));

        DoubleVertex d = a.sum();

        DoubleVertex e = new UniformVertex(0, 10);
        e.setValue(2);

        DoubleVertex f = d.times(e);

        DoubleTensor dfdaForward = f.getDualNumber().getPartialDerivatives().withRespectTo(a);

        PartialDerivatives dfdx = Differentiator.reverseModeAutoDiff(f, a);
        DoubleTensor dfdaReverse = dfdx.withRespectTo(a);

        DoubleTensor expectedDfdx = DoubleTensor.create(new double[]{
            2, 2, 2,
            2, 2, 2
        }, 1, 1, 2, 3);

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

        DoubleVertex f = d.times(e);

        DoubleTensor dfdaForward = f.getDualNumber().getPartialDerivatives().withRespectTo(a);

        PartialDerivatives dfdx = Differentiator.reverseModeAutoDiff(f, a);
        DoubleTensor dfdaReverse = dfdx.withRespectTo(a);

        DoubleTensor expectedDfdx = DoubleTensor.create(new double[]{
            1, 2, 3,
        }, 1, 3, 1, 1);

        assertThat(dfdaForward, equalTo(expectedDfdx));
        assertThat(dfdaReverse, equalTo(expectedDfdx));
    }

    @Test
    public void changesMatchGradient() {
        DoubleVertex inputVertex = new UniformVertex(new int[]{2, 2, 2}, -10.0, 10.0);
        DoubleVertex outputVertex = inputVertex.times(3).sum();

        finiteDifferenceMatchesGradient(ImmutableList.of(inputVertex), outputVertex, 10.0, 1e-10, true);
    }

}
