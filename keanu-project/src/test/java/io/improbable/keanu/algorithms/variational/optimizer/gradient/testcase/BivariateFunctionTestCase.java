package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessAndGradient;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialsOf;
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

        public final FitnessAndGradient evalGradients(double xPoint, double yPoint) {

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

            return new FitnessAndGradient(fEval, gradients);
        }

        public List<? extends Variable> getVariables() {
            return ImmutableList.of(x, y);
        }

        public FitnessFunction getFitnessFunction() {

            return (point) -> evalFunction(
                point.get(x.getReference()).scalar(),
                point.get(y.getReference()).scalar()
            );
        }


        public FitnessFunctionGradient getFitnessFunctionGradient() {

            return new FitnessFunctionGradient() {

                @Override
                public double getFitnessAt(Map<VariableReference, DoubleTensor> point) {
                    return evalFunction(
                        point.get(x.getReference()).scalar(),
                        point.get(y.getReference()).scalar()
                    );
                }

                @Override
                public Map<? extends VariableReference, DoubleTensor> getGradientsAt(Map<VariableReference, DoubleTensor> point) {
                    return evalGradients(
                        point.get(x.getReference()).scalar(),
                        point.get(y.getReference()).scalar()
                    ).getGradients();
                }

                @Override
                public FitnessAndGradient getFitnessAndGradientsAt(Map<VariableReference, DoubleTensor> point) {

                    return evalGradients(
                        point.get(x.getReference()).scalar(),
                        point.get(y.getReference()).scalar()
                    );
                }
            };
        }
    }

    private final BivariateFunction bivariateFunction;

    public BivariateFunctionTestCase(BivariateFunction bivariateFunction) {
        this.bivariateFunction = bivariateFunction;
    }

    @Override
    public FitnessFunction getFitnessFunction() {
        return bivariateFunction.getFitnessFunction();
    }


    @Override
    public FitnessFunctionGradient getFitnessFunctionGradient() {
        return bivariateFunction.getFitnessFunctionGradient();
    }

    @Override
    public List<? extends Variable> getVariables() {
        return bivariateFunction.getVariables();
    }

    @Override
    public void assertResult(OptimizedResult result) {
        double actualX = result.getValueFor(bivariateFunction.x.getReference()).scalar();
        double actualY = result.getValueFor(bivariateFunction.y.getReference()).scalar();

        assertResult(result, actualX, actualY);
    }

    public abstract void assertResult(OptimizedResult result, double actualX, double actualY);

}
