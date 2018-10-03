package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.Diffs.L;
import static io.improbable.keanu.distributions.dual.Diffs.S;
import static io.improbable.keanu.distributions.dual.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import io.improbable.keanu.distributions.continuous.Cauchy;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class CauchyVertex extends DoubleVertex implements ProbabilisticDouble {

  private final DoubleVertex location;
  private final DoubleVertex scale;

  /**
   * One location or scale or both that match a proposed tensor shape of Cauchy
   *
   * <p>If all provided parameters are scalar then the proposed shape determines the shape
   *
   * @param tensorShape the desired shape of the tensor in this vertex
   * @param location the location of the Cauchy with either the same tensorShape as specified for
   *     this vertex or a scalar
   * @param scale the scale of the Cauchy with either the same tensorShape as specified for this
   *     vertex or a scalar
   */
  public CauchyVertex(int[] tensorShape, DoubleVertex location, DoubleVertex scale) {

    checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, location.getShape(), scale.getShape());

    this.location = location;
    this.scale = scale;
    setParents(location, scale);
    setValue(DoubleTensor.placeHolder(tensorShape));
  }

  public CauchyVertex(DoubleVertex location, DoubleVertex scale) {
    this(
        checkHasSingleNonScalarShapeOrAllScalar(location.getShape(), scale.getShape()),
        location,
        scale);
  }

  public CauchyVertex(DoubleVertex location, double scale) {
    this(location, new ConstantDoubleVertex(scale));
  }

  public CauchyVertex(double location, DoubleVertex scale) {
    this(new ConstantDoubleVertex(location), scale);
  }

  public CauchyVertex(double location, double scale) {
    this(new ConstantDoubleVertex(location), new ConstantDoubleVertex(scale));
  }

  public CauchyVertex(int[] tensorShape, DoubleVertex location, double scale) {
    this(tensorShape, location, new ConstantDoubleVertex(scale));
  }

  public CauchyVertex(int[] tensorShape, double location, DoubleVertex scale) {
    this(tensorShape, new ConstantDoubleVertex(location), scale);
  }

  public CauchyVertex(int[] tensorShape, double location, double scale) {
    this(tensorShape, new ConstantDoubleVertex(location), new ConstantDoubleVertex(scale));
  }

  public DoubleVertex getLocation() {
    return location;
  }

  public DoubleVertex getScale() {
    return scale;
  }

  @Override
  public double logProb(DoubleTensor value) {

    DoubleTensor locationValues = location.getValue();
    DoubleTensor scaleValues = scale.getValue();

    DoubleTensor logPdfs = Cauchy.withParameters(locationValues, scaleValues).logProb(value);

    return logPdfs.sum();
  }

  @Override
  public Map<Vertex, DoubleTensor> dLogProb(
      DoubleTensor value, Set<? extends Vertex> withRespectTo) {
    Diffs dlnP = Cauchy.withParameters(location.getValue(), scale.getValue()).dLogProb(value);

    Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

    if (withRespectTo.contains(location)) {
      dLogProbWrtParameters.put(location, dlnP.get(L).getValue());
    }

    if (withRespectTo.contains(scale)) {
      dLogProbWrtParameters.put(scale, dlnP.get(S).getValue());
    }

    if (withRespectTo.contains(this)) {
      dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
    }

    return dLogProbWrtParameters;
  }

  @Override
  public DoubleTensor sample(KeanuRandom random) {
    return Cauchy.withParameters(location.getValue(), scale.getValue()).sample(getShape(), random);
  }
}
