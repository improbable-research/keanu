package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.assertEquals;

public class DualNumbersTest {

    DoubleVertex vA;
    DoubleVertex vB;

    @Before
    public void setup() {
        vA = new GaussianVertex(1.0, 0.0);
        vB = new GaussianVertex(2.0, 0.0);
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
        DoubleVertex vC = vA.plus(vB);
        DoubleVertex vD = vA.minus(vB);
        DoubleVertex vE = vC.multiply(vD);
        assertDiffIsCorrect(vA, vB, vE);
    }

    @Test
    public void diffOverPlusDivideMultiplyLogCombination() {
        DoubleVertex vC = vA.plus(vB);
        DoubleVertex vD = vA.divideBy(vB);
        DoubleVertex vE = vC.multiply(vD);
        assertDiffIsCorrect(vA, vB, new LogVertex(vE));
    }

    private void assertDiffIsCorrect(DoubleVertex vA, DoubleVertex vB, DoubleVertex vC) {

        double A = 1.0;
        double B = 2.0;

        vA.setValue(A);
        vB.setValue(B);
        vC.eval();

        DualNumber cDual = vC.getDualNumber();

        DoubleTensor C = cDual.getValue();
        Map<Long, DoubleTensor> dc = cDual.getPartialDerivatives().asMap();

        double da = 0.00000001;

        vA.setValue(vA.getValue().plus(da));
        vB.setValue(B);
        vC.eval();

        DoubleTensor dcdaApprox = (vC.getValue().minus(C)).div(da);

        assertEquals(dcdaApprox.scalar(), dc.get(vA.getId()).scalar(), 0.00001);

        double db = da;

        vA.setValue(A);
        vB.setValue(B + db);
        vC.eval();

        DoubleTensor dcdbApprox = (vC.getValue().minus(C)).div(db);

        assertEquals(dcdbApprox.scalar(), dc.get(vB.getId()).scalar(), 0.00001);
    }
}
