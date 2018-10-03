package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.Diffs.MU;
import static io.improbable.keanu.distributions.dual.Diffs.S;
import static io.improbable.keanu.distributions.dual.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import io.improbable.keanu.distributions.continuous.Logistic;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class LogisticVertex extends DoubleVertex implements ProbabilisticDouble {

  private final DoubleVertex mu;
  private final DoubleVertex s;

  /**
   * One mu or s or both driving an arbitrarily shaped tensor of Logistic
   *
   * <p>If all provided parameters are scalar then the proposed shape determines the shape
   *
   * @param tensorShape the desired shape of the vertex
   * @param mu the mu (location) of the Logistic with either the same shape as specified for this
   *     vertex or mu scalar
   * @param s the s (scale) of the Logistic with either the same shape as specified for this vertex
   *     or mu scalar
   */
  public LogisticVertex(int[] tensorShape, DoubleVertex mu, DoubleVertex s) {

    checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, mu.getShape(), s.getShape());

    this.mu = mu;
    this.s = s;
    setParents(mu, s);
    setValue(DoubleTensor.placeHolder(tensorShape));
  }

  public LogisticVertex(DoubleVertex mu, DoubleVertex s) {
    this(checkHasSingleNonScalarShapeOrAllScalar(mu.getShape(), s.getShape()), mu, s);
  }

  public LogisticVertex(DoubleVertex mu, double s) {
    this(mu, new ConstantDoubleVertex(s));
  }

  public LogisticVertex(double mu, DoubleVertex s) {
    this(new ConstantDoubleVertex(mu), s);
  }

  public LogisticVertex(double mu, double s) {
    this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(s));
  }

  @Override
  public double logProb(DoubleTensor value) {
    DoubleTensor muValues = mu.getValue();
    DoubleTensor sValues = s.getValue();

    DoubleTensor logPdfs = Logistic.withParameters(muValues, sValues).logProb(value);

    return logPdfs.sum();
  }

  @Override
  public Map<Vertex, DoubleTensor> dLogProb(
      DoubleTensor value, Set<? extends Vertex> withRespectTo) {
    Diffs dlnP = Logistic.withParameters(mu.getValue(), s.getValue()).dLogProb(value);

    Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

    if (withRespectTo.contains(mu)) {
      dLogProbWrtParameters.put(mu, dlnP.get(MU).getValue());
    }

    if (withRespectTo.contains(s)) {
      dLogProbWrtParameters.put(s, dlnP.get(S).getValue());
    }

    if (withRespectTo.contains(this)) {
      dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
    }

    return dLogProbWrtParameters;
  }

  @Override
  public DoubleTensor sample(KeanuRandom random) {
    return Logistic.withParameters(mu.getValue(), s.getValue()).sample(getShape(), random);
  }
}
