package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.Diffs.BETA;
import static io.improbable.keanu.distributions.dual.Diffs.MU;
import static io.improbable.keanu.distributions.dual.Diffs.X;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import io.improbable.keanu.distributions.continuous.Laplace;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class LaplaceVertex extends DoubleVertex implements ProbabilisticDouble {

  private final DoubleVertex mu;
  private final DoubleVertex beta;

  /**
   * One mu or beta or both that match a proposed tensor shape of Laplace
   *
   * <p>If all provided parameters are scalar then the proposed shape determines the shape
   *
   * @param tensorShape the desired shape of the tensor within the vertex
   * @param mu the mu of the Laplace with either the same shape as specified for this vertex or a
   *     scalar
   * @param beta the beta of the Laplace with either the same shape as specified for this vertex or
   *     a scalar
   */
  public LaplaceVertex(int[] tensorShape, DoubleVertex mu, DoubleVertex beta) {

    checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, mu.getShape(), beta.getShape());

    this.mu = mu;
    this.beta = beta;
    setParents(mu, beta);
    setValue(DoubleTensor.placeHolder(tensorShape));
  }

  /**
   * One to one constructor for mapping some shape of mu and sigma to a matching shaped Laplace.
   *
   * @param mu the mu of the Laplace with either the same shape as specified for this vertex or a
   *     scalar
   * @param beta the beta of the Laplace with either the same shape as specified for this vertex or
   *     a scalar
   */
  public LaplaceVertex(DoubleVertex mu, DoubleVertex beta) {
    this(checkHasSingleNonScalarShapeOrAllScalar(mu.getShape(), beta.getShape()), mu, beta);
  }

  public LaplaceVertex(DoubleVertex mu, double beta) {
    this(mu, new ConstantDoubleVertex(beta));
  }

  public LaplaceVertex(double mu, DoubleVertex beta) {
    this(new ConstantDoubleVertex(mu), beta);
  }

  public LaplaceVertex(double mu, double beta) {
    this(new ConstantDoubleVertex(mu), new ConstantDoubleVertex(beta));
  }

  @Override
  public double logProb(DoubleTensor value) {

    DoubleTensor muValues = mu.getValue();
    DoubleTensor betaValues = beta.getValue();

    DoubleTensor logPdfs = Laplace.withParameters(muValues, betaValues).logProb(value);

    return logPdfs.sum();
  }

  @Override
  public Map<Vertex, DoubleTensor> dLogProb(
      DoubleTensor value, Set<? extends Vertex> withRespectTo) {
    Diffs dlnP = Laplace.withParameters(mu.getValue(), beta.getValue()).dLogProb(value);

    Map<Vertex, DoubleTensor> dLogProbWrtParameters = new HashMap<>();

    if (withRespectTo.contains(mu)) {
      dLogProbWrtParameters.put(mu, dlnP.get(MU).getValue());
    }

    if (withRespectTo.contains(beta)) {
      dLogProbWrtParameters.put(beta, dlnP.get(BETA).getValue());
    }

    if (withRespectTo.contains(this)) {
      dLogProbWrtParameters.put(this, dlnP.get(X).getValue());
    }

    return dLogProbWrtParameters;
  }

  @Override
  public DoubleTensor sample(KeanuRandom random) {
    return Laplace.withParameters(mu.getValue(), beta.getValue()).sample(getShape(), random);
  }
}
