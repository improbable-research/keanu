package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import java.util.Map;

public abstract class DoubleBinaryOpVertex extends DoubleVertex
    implements NonProbabilistic<DoubleTensor> {

  protected final DoubleVertex left;
  protected final DoubleVertex right;

  /**
   * A vertex that performs a user defined operation on two vertices
   *
   * @param left a vertex
   * @param right a vertex
   */
  public DoubleBinaryOpVertex(DoubleVertex left, DoubleVertex right) {
    this(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right);
  }

  /**
   * A vertex that performs a user defined operation on two vertices
   *
   * @param shape the shape of the resulting vertex
   * @param left a vertex
   * @param right a vertex
   */
  public DoubleBinaryOpVertex(int[] shape, DoubleVertex left, DoubleVertex right) {
    this.left = left;
    this.right = right;
    setParents(left, right);
    setValue(DoubleTensor.placeHolder(shape));
  }

  @Override
  public DoubleTensor sample(KeanuRandom random) {
    return op(left.sample(random), right.sample(random));
  }

  @Override
  public DoubleTensor calculate() {
    return op(left.getValue(), right.getValue());
  }

  public DoubleVertex getLeft() {
    return left;
  }

  public DoubleVertex getRight() {
    return right;
  }

  @Override
  public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
    try {
      return dualOp(dualNumbers.get(left), dualNumbers.get(right));
    } catch (UnsupportedOperationException e) {
      return super.calculateDualNumber(dualNumbers);
    }
  }

  protected abstract DoubleTensor op(DoubleTensor l, DoubleTensor r);

  protected abstract DualNumber dualOp(DualNumber l, DualNumber r);
}
