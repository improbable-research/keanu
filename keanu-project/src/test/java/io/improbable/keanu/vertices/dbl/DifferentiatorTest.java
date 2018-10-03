package io.improbable.keanu.vertices.dbl;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesGradient;
import static org.junit.Assert.assertEquals;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import org.junit.Test;

public class DifferentiatorTest {

  @Test
  public void canForwardAutoDiffOfSingleOuputWithRespectToMany() {

    DoubleVertex A = new GaussianVertex(0, 1);
    DoubleVertex B = new GaussianVertex(0, 1);
    DoubleVertex C = A.times(B);

    PartialDerivatives dC = C.getDualNumber().getPartialDerivatives();

    DoubleTensor dCdA = dC.withRespectTo(A);
    DoubleTensor dCdB = dC.withRespectTo(B);

    assertEquals(A.getValue().reshape(1, 1, 1, 1), dCdB);
    assertEquals(B.getValue().reshape(1, 1, 1, 1), dCdA);
  }

  @Test
  public void canReverseAutoDiffOfMultiplicationWithSingleOutputWithRespectToMany() {

    DoubleVertex A = new GaussianVertex(0, 1);
    DoubleVertex B = new GaussianVertex(0, 1);
    DoubleVertex C = A.times(B);

    PartialDerivatives dC = Differentiator.reverseModeAutoDiff(C, ImmutableSet.of(A, B));

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

    PartialDerivatives dY = Differentiator.reverseModeAutoDiff(Y, ImmutableSet.of(A, B));

    DoubleTensor dYdA = dY.withRespectTo(A);
    DoubleTensor dYdB = dY.withRespectTo(B);

    assertEquals(A.getValue().reciprocal().scalar(), dYdA.scalar(), 1e-5);
    assertEquals(B.getValue().reciprocal().scalar(), dYdB.scalar(), 1e-5);
  }

  @Test
  public void reverseAutoDiffMatchesForwardWithSingleOutputWithRespectToMany() {

    int[] shape = new int[] {2, 2};
    DoubleVertex A = new GaussianVertex(shape, 0, 1);
    A.setValue(DoubleTensor.linspace(0.1, 2, 4).reshape(shape));
    DoubleVertex B = new GaussianVertex(shape, 0, 1);
    B.setValue(DoubleTensor.linspace(0.2, 1, 4).reshape(shape));
    DoubleVertex D = A.atan2(B).sigmoid().times(B);
    DoubleVertex C = A.sin().cos().div(D);
    DoubleVertex E = C.times(D).pow(A).acos();
    DoubleVertex G = E.log().tan().asin().atan();
    DoubleVertex F = D.plus(B).exp();
    DoubleVertex H = G.plus(F).sum();

    PartialDerivatives dHReverse = Differentiator.reverseModeAutoDiff(H, ImmutableSet.of(A, B));
    PartialDerivatives dHForward = H.getDualNumber().getPartialDerivatives();

    DoubleTensor dHdAReverse = dHReverse.withRespectTo(A);
    DoubleTensor dHdBReverse = dHReverse.withRespectTo(B);

    DoubleTensor dHdAForward = dHForward.withRespectTo(A);
    DoubleTensor dHdBForward = dHForward.withRespectTo(B);

    assertEquals(dHdAReverse, dHdAForward);
    assertEquals(dHdBReverse, dHdBForward);
  }

  @Test
  public void reverseAutoDiffOfRank3MatchesForwardWithSingleOutputWithRespectToMany() {

    int[] shape = new int[] {2, 2, 2};
    DoubleVertex A = new GaussianVertex(shape, 0, 1);
    A.setValue(DoubleTensor.linspace(0.1, 2, (int) TensorShape.getLength(shape)).reshape(shape));
    DoubleVertex B = new GaussianVertex(shape, 0, 1);
    B.setValue(DoubleTensor.linspace(0.2, 1, (int) TensorShape.getLength(shape)).reshape(shape));

    DoubleVertex C = new GaussianVertex(shape, 0, 1);
    C.setValue(DoubleTensor.linspace(0.2, 0.8, (int) TensorShape.getLength(shape)).reshape(shape));

    DoubleVertex D = A.atan2(B).sigmoid().times(B);
    DoubleVertex J = A.sin().cos().div(D);
    DoubleVertex E = J.times(D).pow(A).acos();
    DoubleVertex G = E.log().tan().atan();
    DoubleVertex F = D.plus(B).exp();
    DoubleVertex H = G.plus(F).sum().times(A).sum().times(C);

    finiteDifferenceMatchesGradient(ImmutableList.of(A, B, C), H, 0.001, 1e-3, true);
  }

  @Test
  public void
      canReverseAutoDiffOfMultiplicationLogSinAndSumWithSingleConditionalOutputWithRespectToMany() {

    DoubleVertex A = new GaussianVertex(new int[] {2, 2}, 0, 1);
    A.setValue(DoubleTensor.create(3.0, new int[] {2, 2}));
    DoubleVertex B = new GaussianVertex(new int[] {2, 2}, 0, 1);
    B.setValue(DoubleTensor.create(5.0, new int[] {2, 2}));
    DoubleVertex D = A.times(B);
    DoubleVertex C = A.sin();
    DoubleVertex E = C.times(D);
    DoubleVertex G = E.log();
    DoubleVertex F = D.plus(B);

    BoolVertex predicate =
        ConstantVertex.of(
            BooleanTensor.create(new boolean[] {true, false, true, false}, new int[] {2, 2}));
    DoubleVertex H = If.isTrue(predicate).then(G).orElse(F);

    PartialDerivatives dH = Differentiator.reverseModeAutoDiff(H, ImmutableSet.of(A, B));

    DoubleTensor dHdA = dH.withRespectTo(A);
    DoubleTensor dHdB = dH.withRespectTo(B);

    DoubleTensor predicateTrueMask = predicate.getValue().toDoubleMask();
    DoubleTensor predicateFalseMask = predicate.getValue().not().toDoubleMask();
    DoubleTensor AValue = A.getValue();
    DoubleTensor BValue = B.getValue();

    DoubleTensor expecteddHdA =
        AValue.reciprocal()
            .plus(AValue.cos().div(AValue.sin()))
            .times(predicateTrueMask)
            .plus(BValue.times(predicateFalseMask))
            .reshape(1, 4)
            .diag()
            .reshape(2, 2, 2, 2);

    DoubleTensor expecteddHdB =
        BValue.reciprocal()
            .times(predicateTrueMask)
            .plus(AValue.plus(1).times(predicateFalseMask))
            .reshape(1, 4)
            .diag()
            .reshape(2, 2, 2, 2);

    assertEquals(expecteddHdA, dHdA);
    assertEquals(expecteddHdB, dHdB);
  }
}
