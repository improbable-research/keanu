package io.improbable.keanu.vertices;

import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.*;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class CovarianceTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Test
    public void youCanGetTheCovarianceMatrix() {
        VertexId id1 = new VertexId();
        VertexId id2 = new VertexId();
        DoubleTensor matrix = DoubleTensor.eye(2);
        Covariance covariance = new Covariance(matrix, id1, id2);
        assertThat(covariance.asTensor(), equalTo(matrix));
    }

    @Test
    public void youMustUseARankTwoMatrixToConstructIt() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("The covariance matrix must be rank 2");

        new Covariance(DoubleTensor.arange(0., 8.).reshape(2, 2, 2), new VertexId(), new VertexId());
    }

    @Test
    public void youMustUseASquareMatrixToConstructIt() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("The covariance matrix must be square");

        new Covariance(DoubleTensor.create(1., 2., 3.), new VertexId(), new VertexId());
    }

    @Test
    public void theNumberOfVertexIdsMustMatchTheLengthOfTheMatrix() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("You must pass in 2 VertexIds, to match the dimension of the matrix");

        new Covariance(DoubleTensor.eye(2), new VertexId());
    }

    @Test
    public void theMatrixMustBeSymmetric() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("The covariance matrix must be symmetric");

        new Covariance(DoubleTensor.arange(0., 4.).reshape(2, 2), new VertexId(), new VertexId());
    }

    @Test
    public void youCanGetASubsetOfTheCovarianceMatrix() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("The covariance matrix must be symmetric");
        VertexId id1 = new VertexId();
        VertexId id2 = new VertexId();
        VertexId id3 = new VertexId();
        DoubleTensor matrix = DoubleTensor.arange(1., 10.).reshape(3, 3);
        Covariance covariance = new Covariance(matrix, id1, id2, id3);
        assertThat(covariance.getSubMatrix(id1, id3), equalTo(DoubleTensor.create(1., 3., 7., 9.).reshape(2, 2)));
    }
}