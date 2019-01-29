package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.algorithms.variational.optimizer.*;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialsOf;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static junit.framework.TestCase.assertEquals;

/**
 * This is a common test case for gradient optimization algorithms.
 * <p>
 * https://en.wikipedia.org/wiki/Rosenbrock_function
 * <p>
 * The function has been inverted here because our optimizers find maxima and this
 * function provides a global minima. The global maxima is at (a, a^2) so for a=1,b=100
 * the maxima will be (1,1)
 */
public class RosenbrockTestCase implements GradientOptimizationAlgorithmTestCase {

    private final DoubleVertex a;
    private final DoubleVertex b;

    private final DoubleVertex x;
    private final DoubleVertex y;

    private final DoubleVertex f;

    private final double expectedX, expectedY;

    public RosenbrockTestCase(double aParameter, double bParameter, double expectedX, double expectedY) {
        this.expectedX = expectedX;
        this.expectedY = expectedY;

        this.a = new ConstantDoubleVertex(aParameter);
        this.b = new ConstantDoubleVertex(bParameter);

        x = new UniformVertex(-1000, 1000);
        y = new UniformVertex(-1000, 1000);

        x.setValue(0);
        y.setValue(3);

        f = a.minus(x).pow(2).plus(
            b.times(y.minus(x.pow(2)).pow(2))
        ).times(-1);
    }

    @Override
    public FitnessFunction getFitnessFunction() {

        return (point) -> evalFunction(
            point.get(x.getReference()).scalar(),
            point.get(y.getReference()).scalar()
        );
    }

    private final double evalFunction(double xPoint, double yPoint) {
        x.setValue(xPoint);
        y.setValue(yPoint);

        return f.eval().scalar();
    }

    @Override
    public FitnessFunctionGradient getFitnessFunctionGradient() {

        return (point) -> {

            double xPoint = point.get(x.getReference()).scalar();
            double yPoint = point.get(y.getReference()).scalar();

            double fEval = evalFunction(
                xPoint,
                yPoint
            );

            PartialsOf partialsOf = Differentiator.reverseModeAutoDiff(f, x, y);

            Map<VariableReference, DoubleTensor> gradients = new HashMap<>();

            DoubleTensor dfdx = partialsOf.withRespectTo(x);
            DoubleTensor dfdy = partialsOf.withRespectTo(y);

            gradients.put(x.getReference(), dfdx);
            gradients.put(y.getReference(), dfdy);

            return gradients;
        };
    }

    @Override
    public List<? extends Variable> getVariables() {
        return ImmutableList.of(x, y);
    }

    @Override
    public void assertResult(OptimizedResult result) {
        Double actualX = result.get(x.getReference()).scalar();
        Double actualY = result.get(y.getReference()).scalar();

        assertEquals(expectedX, actualX, 1e-2);
        assertEquals(expectedY, actualY, 1e-2);
    }
}
