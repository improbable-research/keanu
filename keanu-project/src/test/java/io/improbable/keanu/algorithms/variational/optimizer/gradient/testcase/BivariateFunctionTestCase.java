package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialsOf;
import lombok.Value;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class BivariateFunctionTestCase implements GradientOptimizationAlgorithmTestCase {

    @Value
    public static class BivariateFunction {
        private final DoubleVertex x;
        private final DoubleVertex y;
        private final DoubleVertex f;

        public final double evalFunction(double xPoint, double yPoint) {
            x.setValue(xPoint);
            y.setValue(yPoint);

            return f.eval().scalar();
        }

        public final Map<VariableReference, DoubleTensor> evalGradients(double xPoint, double yPoint) {

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
        }

        public List<? extends Variable> getVariables() {
            return ImmutableList.of(x, y);
        }
    }

    private final BivariateFunction bivariateFunction;

    public BivariateFunctionTestCase(BivariateFunction bivariateFunction) {
        this.bivariateFunction = bivariateFunction;
    }

    @Override
    public FitnessFunction getFitnessFunction() {

        return (point) -> bivariateFunction.evalFunction(
            point.get(bivariateFunction.x.getReference()).scalar(),
            point.get(bivariateFunction.y.getReference()).scalar()
        );
    }


    @Override
    public FitnessFunctionGradient getFitnessFunctionGradient() {

        return (point) -> bivariateFunction.evalGradients(
            point.get(bivariateFunction.x.getReference()).scalar(),
            point.get(bivariateFunction.y.getReference()).scalar()
        );
    }

    @Override
    public List<? extends Variable> getVariables() {
        return bivariateFunction.getVariables();
    }

    @Override
    public void assertResult(OptimizedResult result) {
        Double actualX = result.get(bivariateFunction.x.getReference()).scalar();
        Double actualY = result.get(bivariateFunction.y.getReference()).scalar();

        assertResult(result, actualX, actualY);
    }

    public abstract void assertResult(OptimizedResult result, Double actualX, Double actualY);

}
