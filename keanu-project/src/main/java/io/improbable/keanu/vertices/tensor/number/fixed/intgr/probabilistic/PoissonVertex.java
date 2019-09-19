package io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.discrete.Poisson;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadShape;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphSupplier;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerPlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertexWrapper.wrapIfNeeded;

public class PoissonVertex extends VertexImpl<IntegerTensor, IntegerVertex>
    implements ProbabilisticInteger, LogProbGraphSupplier {

    private final DoubleVertex mu;
    private static final String MU_NAME = "mu";

    /**
     * Poisson with mu as a hyperparameter. Mu here is aka lambda.
     *
     * @param shape the desired shape of the vertex. This must be broadcastable with the mu shape.
     * @param mu    the mu of the Poisson. The shape of mu must be broadcastable with shape.
     */
    public PoissonVertex(@LoadShape long[] shape,
                         @LoadVertexParam(MU_NAME) Vertex<DoubleTensor, ?> mu) {
        super(TensorShape.getBroadcastResultShape(shape, mu.getShape()));

        this.mu = wrapIfNeeded(mu);
        setParents(mu);
    }

    public PoissonVertex(long[] shape, double mu) {
        this(shape, new ConstantDoubleVertex(mu));
    }

    /**
     * One to one constructor for mapping some shape of mu to
     * a matching shaped Poisson.
     *
     * @param mu mu with same shape as desired Poisson tensor or scalar
     */
    @ExportVertexToPythonBindings
    public PoissonVertex(Vertex<DoubleTensor, ?> mu) {
        this(mu.getShape(), mu);
    }

    public PoissonVertex(double mu) {
        this(Tensor.SCALAR_SHAPE, new ConstantDoubleVertex(mu));
    }

    @SaveVertexParam(MU_NAME)
    public DoubleVertex getMu() {
        return mu;
    }

    @Override
    public double logProb(IntegerTensor value) {
        return Poisson.withParameters(mu.getValue()).logProb(value).sumNumber();
    }

    @Override
    public LogProbGraph logProbGraph() {
        IntegerPlaceholderVertex valuePlaceholder = new IntegerPlaceholderVertex(this.getShape());
        DoublePlaceholderVertex muPlaceholder = new DoublePlaceholderVertex(mu.getShape());

        return LogProbGraph.builder()
            .input(this, valuePlaceholder)
            .input(mu, muPlaceholder)
            .output(Poisson.logProbOutput(valuePlaceholder, muPlaceholder))
            .build();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(IntegerTensor value, Set<? extends Vertex> withRespectTo) {
        boolean wrtMu = withRespectTo.contains(mu);

        DoubleTensor[] result = Poisson.withParameters(mu.getValue()).dLogProb(value, wrtMu);
        if (wrtMu) {
            return Collections.singletonMap(mu, result[0]);
        } else {
            return Collections.emptyMap();
        }
    }

    @Override
    public IntegerTensor sample(long[] shape, KeanuRandom random) {
        return Poisson.withParameters(mu.getValue()).sample(shape, random);
    }
}
