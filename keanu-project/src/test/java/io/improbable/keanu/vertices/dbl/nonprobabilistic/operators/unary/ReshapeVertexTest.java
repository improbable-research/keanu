package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import org.junit.Assert;
import org.junit.Test;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;

public class ReshapeVertexTest {

    @Test
    public void reshapeVertex() {
        DoubleVertex a = VertexOfType.uniform(0., 10.);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex reshapeVertex = a.reshape(4, 1);
        reshapeVertex.getValue();

        Assert.assertArrayEquals(new int[]{4, 1}, reshapeVertex.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4}, reshapeVertex.getValue().asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void reshapeCorrectlyReshapesPartialDerivative() {
        DoubleVertex m = VertexOfType.uniform(0., 10.);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex alpha = VertexOfType.uniform(0., 10.);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex N = m.matrixMultiply(alpha);

        DoubleVertex reshapedN = N.reshape(4, 1);
        DoubleTensor reshapedPartial = reshapedN.getDualNumber().getPartialDerivatives().withRespectTo(m);

        Assert.assertArrayEquals(new int[]{4, 1, 2, 2}, reshapedPartial.getShape());
    }

    @Test
    public void flatPartialDerivativeIsTheSameAfterReshape() {
        DoubleVertex m = VertexOfType.uniform(0., 10.);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex a = VertexOfType.uniform(0., 10.);
        a.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex N = m.matrixMultiply(a);
        DualNumber NDual = N.getDualNumber();

        DoubleTensor dNdm = NDual.getPartialDerivatives().withRespectTo(m);
        DoubleTensor dNda = NDual.getPartialDerivatives().withRespectTo(a);

        double[] nWrtMpartialsBeforeReshape = dNdm.asFlatDoubleArray();
        double[] nWrtApartialsBeforeReshape = dNda.asFlatDoubleArray();

        DoubleVertex reshapedN = N.reshape(4, 1);
        DoubleTensor reshapedPartialWrtM = reshapedN.getDualNumber().getPartialDerivatives().withRespectTo(m);
        DoubleTensor reshapedPartialWrtA = reshapedN.getDualNumber().getPartialDerivatives().withRespectTo(a);

        Assert.assertArrayEquals(nWrtMpartialsBeforeReshape, reshapedPartialWrtM.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(nWrtApartialsBeforeReshape, reshapedPartialWrtA.asFlatDoubleArray(), 1e-6);
    }

}
