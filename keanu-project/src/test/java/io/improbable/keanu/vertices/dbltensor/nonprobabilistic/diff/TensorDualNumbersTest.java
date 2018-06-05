package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary.TensorLogVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.assertEquals;

public class TensorDualNumbersTest {

    DoubleTensorVertex vA;
    DoubleTensorVertex vB;

    @Before
    public void setup() {
        vA = new TensorGaussianVertex(1.0, 0.0);
        vB = new TensorGaussianVertex(2.0, 0.0);
    }

    @Test
    public void diffOverMultiply() {
        assertDiffIsCorrect(vA, vB, vA.multiply(vB));
    }

    @Test
    public void diffOverAddition() {
        assertDiffIsCorrect(vA, vB, vA.plus(vB));
    }

    @Test
    public void diffOverSubtraction() {
        assertDiffIsCorrect(vA, vB, vA.minus(vB));
    }

    @Test
    public void diffOverExponent() {
        assertDiffIsCorrect(vA, vB, vA.multiply(vB).exp());
    }

    @Test
    public void diffOverPlusMinusMultiplyCombination() {
        DoubleTensorVertex vC = vA.plus(vB);
        DoubleTensorVertex vD = vA.minus(vB);
        DoubleTensorVertex vE = vC.multiply(vD);
        assertDiffIsCorrect(vA, vB, vE);
    }

    @Test
    public void diffOverPlusDivideMultiplyLogCombination() {
        DoubleTensorVertex vC = vA.plus(vB);
        DoubleTensorVertex vD = vA.divideBy(vB);
        DoubleTensorVertex vE = vC.multiply(vD);
        assertDiffIsCorrect(vA, vB, new TensorLogVertex(vE));
    }

    private void assertDiffIsCorrect(DoubleTensorVertex vA, DoubleTensorVertex vB, DoubleTensorVertex vC) {

        double A = 1.0;
        double B = 2.0;

        vA.setValue(A);
        vB.setValue(B);
        vC.lazyEval();

        TensorDualNumber cDual = vC.getDualNumber();

        DoubleTensor C = cDual.getValue();
        Map<Long, DoubleTensor> dc = cDual.getPartialDerivatives().asMap();

        double da = 0.00000001;

        vA.setValue(vA.getValue().plus(da));
        vB.setValue(B);
        vC.lazyEval();

        DoubleTensor dcdaApprox = (vC.getValue().minus(C)).div(da);

        assertEquals(dcdaApprox.scalar(), dc.get(vA.getId()).scalar(), 0.00001);

        double db = da;

        vA.setValue(A);
        vB.setValue(B + db);
        vC.lazyEval();

        DoubleTensor dcdbApprox = (vC.getValue().minus(C)).div(db);

        assertEquals(dcdbApprox.scalar(), dc.get(vB.getId()).scalar(), 0.00001);
    }
}
