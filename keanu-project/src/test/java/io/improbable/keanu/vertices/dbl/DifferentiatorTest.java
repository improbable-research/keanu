package io.improbable.keanu.vertices.dbl;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialsOf;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SumVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesWithinEpsilonAndShapesMatch;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertEquals;

public class DifferentiatorTest {

    @Test
    public void canForwardAutoDiffOfSingleOuputWithRespectToMany() {

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);
        MultiplicationVertex C = A.times(B);

        DoubleTensor dCdA = Differentiator.forwardModeAutoDiff(A, C).of(C);
        DoubleTensor dCdB = Differentiator.forwardModeAutoDiff(B, C).of(C);

        assertEquals(A.getValue(), dCdB);
        assertEquals(B.getValue(), dCdA);
    }

    @Test
    public void canReverseAutoDiffOfMultiplicationWithSingleOutputWithRespectToMany() {

        DoubleVertex A = new GaussianVertex(0, 1);
        DoubleVertex B = new GaussianVertex(0, 1);
        DoubleVertex C = A.times(B);

        PartialsOf dC = Differentiator.reverseModeAutoDiff(C, ImmutableSet.of(A, B));

        DoubleTensor dCdA = dC.withRespectTo(A);
        DoubleTensor dCdB = dC.withRespectTo(B);

        assertEquals(A.getValue().scalar(), dCdB.scalar(), 1e-5);
        assertEquals(B.getValue().scalar(), dCdA.scalar(), 1e-5);
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

        PartialsOf dY = Differentiator.reverseModeAutoDiff(Y, ImmutableSet.of(A, B));

        DoubleTensor dYdA = dY.withRespectTo(A);
        DoubleTensor dYdB = dY.withRespectTo(B);

        assertEquals(A.getValue().reciprocal().scalar(), dYdA.scalar(), 1e-5);
        assertEquals(B.getValue().reciprocal().scalar(), dYdB.scalar(), 1e-5);
    }

    @Test
    public void reverseAutoDiffMatchesForwardWithSingleOutputWithRespectToMany() {

        long[] shape = new long[]{2, 2};
        GaussianVertex A = new GaussianVertex(shape, 0, 1);
        A.setValue(DoubleTensor.linspace(0.1, 2, 4).reshape(shape));
        GaussianVertex B = new GaussianVertex(shape, 0, 1);
        B.setValue(DoubleTensor.linspace(0.2, 1, 4).reshape(shape));
        DoubleVertex D = A.atan2(B).sigmoid().times(B);
        DoubleVertex C = A.sin().cos().div(D);
        DoubleVertex E = C.times(D).pow(A).acos();
        DoubleVertex G = E.log().tan().asin().atan();
        DoubleVertex F = D.plus(B).exp();
        SumVertex H = G.plus(F).sum();

        PartialsOf dHReverse = Differentiator.reverseModeAutoDiff(H, ImmutableSet.of(A, B));

        DoubleTensor dHdAReverse = dHReverse.withRespectTo(A);
        DoubleTensor dHdBReverse = dHReverse.withRespectTo(B);

        DoubleTensor dHdAForward = Differentiator.forwardModeAutoDiff(A, H).of(H);
        DoubleTensor dHdBForward = Differentiator.forwardModeAutoDiff(B, H).of(H);

        assertThat(dHdAReverse, valuesWithinEpsilonAndShapesMatch(dHdAForward, 1e-8));
        assertThat(dHdBReverse, valuesWithinEpsilonAndShapesMatch(dHdBForward, 1e-8));
    }

    @Test
    public void reverseAutoDiffOfRank3MatchesForwardWithSingleOutputWithRespectToMany() {

        long[] shape = new long[]{2, 2, 2};
        GaussianVertex A = new GaussianVertex(shape, 0, 1);
        A.setValue(DoubleTensor.linspace(0.1, 2, (int) TensorShape.getLength(shape)).reshape(shape));
        GaussianVertex B = new GaussianVertex(shape, 0, 1);
        B.setValue(DoubleTensor.linspace(0.2, 1, (int) TensorShape.getLength(shape)).reshape(shape));

        GaussianVertex C = new GaussianVertex(shape, 0, 1);
        C.setValue(DoubleTensor.linspace(0.2, 0.8, (int) TensorShape.getLength(shape)).reshape(shape));

        DoubleVertex D = A.atan2(B).sigmoid().times(B);
        DoubleVertex J = A.sin().cos().div(D);
        DoubleVertex E = J.times(D).pow(A).acos();
        DoubleVertex G = E.log().tan().atan();
        DoubleVertex F = D.plus(B).exp();
        MultiplicationVertex H = G.plus(F).sum().times(A).sum().times(C);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(A, B, C), H, 0.001, 1e-3);
    }

    @Test
    public void canReverseAutoDiffOfMultiplicationLogSinAndSumWithSingleConditionalOutputWithRespectToMany() {

        DoubleVertex A = new GaussianVertex(new long[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(3.0, new long[]{2, 2}));
        DoubleVertex B = new GaussianVertex(new long[]{2, 2}, 0, 1);
        B.setValue(DoubleTensor.create(5.0, new long[]{2, 2}));
        DoubleVertex D = A.times(B);
        DoubleVertex C = A.sin();
        DoubleVertex E = C.times(D);
        DoubleVertex G = E.log();
        DoubleVertex F = D.plus(B);

        BooleanVertex predicate = ConstantVertex.of(BooleanTensor.create(new boolean[]{true, false, true, false}, new long[]{2, 2}));
        DoubleVertex H = If.isTrue(predicate).then(G).orElse(F);

        PartialsOf dH = Differentiator.reverseModeAutoDiff(H, ImmutableSet.of(A, B));

        DoubleTensor dHdA = dH.withRespectTo(A);
        DoubleTensor dHdB = dH.withRespectTo(B);

        DoubleTensor predicateTrueMask = predicate.getValue().toDoubleMask();
        DoubleTensor predicateFalseMask = predicate.getValue().not().toDoubleMask();
        DoubleTensor AValue = A.getValue();
        DoubleTensor BValue = B.getValue();

        DoubleTensor expecteddHdA = AValue.reciprocal().plus(AValue.cos().div(AValue.sin())).times(predicateTrueMask)
            .plus(BValue.times(predicateFalseMask)).reshape(1, 4).diag().reshape(2, 2, 2, 2);

        DoubleTensor expecteddHdB = BValue.reciprocal().times(predicateTrueMask).plus(AValue.plus(1).times(predicateFalseMask)).reshape(1, 4).diag().reshape(2, 2, 2, 2);

        assertEquals(expecteddHdA, dHdA);
        assertEquals(expecteddHdB, dHdB);
    }

}
