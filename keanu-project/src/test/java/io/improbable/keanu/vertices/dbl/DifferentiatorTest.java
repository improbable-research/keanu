package io.improbable.keanu.vertices.dbl;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class DifferentiatorTest {

    @Test
    public void canForwardAutoDiffOfSingleOuputWithRespectToMany() {

        DoubleVertex A = new GaussianVertex(0, 1);
        DoubleVertex B = new GaussianVertex(0, 1);
        DoubleVertex C = A.times(B);

        PartialDerivatives dC = Differentiator.forwardModeAutoDiff(C, Arrays.asList(A, B));

        DoubleTensor dCdA = dC.withRespectTo(A);
        DoubleTensor dCdB = dC.withRespectTo(B);

        assertEquals(A.getValue(), dCdB);
        assertEquals(B.getValue(), dCdA);
    }

    @Test
    public void canReverseAutoDiffOfMultiplicationWithSingleOutputWithRespectToMany() {

        DoubleVertex A = new GaussianVertex(0, 1);
        DoubleVertex B = new GaussianVertex(0, 1);
        DoubleVertex C = A.times(B);

        PartialDerivatives dC = Differentiator.reverseModeAutoDiff(C, ImmutableSet.of(A, B));

        DoubleTensor dCdA = dC.withRespectTo(A);
        DoubleTensor dCdB = dC.withRespectTo(B);

        assertEquals(A.getValue(), dCdB);
        assertEquals(B.getValue(), dCdA);
    }

    @Test
    public void canReverseAutoDiffOfMultiplicationAndLogWithSingleOutputWithRespectToMany() {

        DoubleVertex A = new GaussianVertex(0, 1);
        A.setValue(3.0);
        DoubleVertex B = new GaussianVertex(0, 1);
        B.setValue(5.0);

        DoubleVertex C = A.times(B);
        DoubleVertex E = C.times(2);
        DoubleVertex Y = E.log();

        PartialDerivatives dY = Differentiator.reverseModeAutoDiff(Y, ImmutableSet.of(A, B));

        DoubleTensor dYdA = dY.withRespectTo(A);
        DoubleTensor dYdB = dY.withRespectTo(B);

        assertEquals(A.getValue().reciprocal().scalar(), dYdA.scalar(), 1e-5);
        assertEquals(B.getValue().reciprocal().scalar(), dYdB.scalar(), 1e-5);
    }

    @Test
    public void canReverseAutoDiffOfMultiplicationLogSinAndSumWithSingleOutputWithRespectToMany() {

        DoubleVertex A = new GaussianVertex(new int[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(3.0, new int[]{2, 2}));
        DoubleVertex B = new GaussianVertex(new int[]{2, 2}, 0, 1);
        B.setValue(DoubleTensor.create(5.0, new int[]{2, 2}));
        DoubleVertex D = A.times(B);
        DoubleVertex C = A.sin();
        DoubleVertex E = C.times(D);
        DoubleVertex G = E.log();
        DoubleVertex F = D.plus(B);
        DoubleVertex H = G.plus(F);

        PartialDerivatives dH = Differentiator.reverseModeAutoDiff(H, ImmutableSet.of(A, B));

        DoubleTensor dHdA = dH.withRespectTo(A);
        DoubleTensor dHdB = dH.withRespectTo(B);

        DoubleTensor AValue = A.getValue();
        DoubleTensor BValue = B.getValue();

        DoubleTensor expecteddHdA = AValue.reciprocal().plus(AValue.cos().div(AValue.sin())).plus(BValue).reshape(1, 4).diag().reshape(2, 2, 2, 2);
        DoubleTensor expecteddHdB = BValue.reciprocal().plus(1).plus(AValue).reshape(1, 4).diag().reshape(2, 2, 2, 2);

        System.out.println("dhda shape " + Arrays.toString(dHdA.getShape()));
        System.out.println("dhdb shape " + Arrays.toString(dHdB.getShape()));

        assertEquals(expecteddHdA, dHdA);
        assertEquals(expecteddHdB, dHdB);
    }

    @Test
    public void canReverseAutoDiffOfMultiplicationLogSinAndSumWithSingleConditionalOutputWithRespectToMany() {

        DoubleVertex A = new GaussianVertex(new int[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(3.0, new int[]{2, 2}));
        DoubleVertex B = new GaussianVertex(new int[]{2, 2}, 0, 1);
        B.setValue(DoubleTensor.create(5.0, new int[]{2, 2}));
        DoubleVertex D = A.times(B);
        DoubleVertex C = A.sin();
        DoubleVertex E = C.times(D);
        DoubleVertex G = E.log();
        DoubleVertex F = D.plus(B);

        BoolVertex predicate = ConstantVertex.of(BooleanTensor.create(new boolean[]{true, false, true, false}, new int[]{2, 2}));
        DoubleVertex H = If.isTrue(predicate).then(G).orElse(F);

        PartialDerivatives dH = Differentiator.reverseModeAutoDiff(H, ImmutableSet.of(A, B));

        DoubleTensor dHdA = dH.withRespectTo(A);
        DoubleTensor dHdB = dH.withRespectTo(B);

        DoubleTensor predicateTrueMask = predicate.getValue().toDoubleMask();
        DoubleTensor predicateFalseMask = predicate.getValue().not().toDoubleMask();
        DoubleTensor AValue = A.getValue();
        DoubleTensor BValue = B.getValue();

        DoubleTensor expecteddHdA = AValue.reciprocal().plus(AValue.cos().div(AValue.sin())).times(predicateTrueMask)
            .plus(BValue.times(predicateFalseMask)).reshape(1, 4).diag().reshape(2, 2, 2, 2);

        DoubleTensor expecteddHdB = BValue.reciprocal().times(predicateTrueMask).plus(AValue.plus(1).times(predicateFalseMask)).reshape(1, 4).diag().reshape(2, 2, 2, 2);

        System.out.println("dhda shape " + Arrays.toString(dHdA.getShape()));
        System.out.println("dhdb shape " + Arrays.toString(dHdB.getShape()));
        assertEquals(expecteddHdA, dHdA);
        assertEquals(expecteddHdB, dHdB);
    }
}
