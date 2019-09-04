package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import com.google.common.primitives.Ints;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.getAsDoubleTensors;

/**
 * https://github.com/miikama/cpp-math/blob/master/CppNumericalSolvers/include/cppoptlib/solver/lbfgssolver.h
 */
public class LBFGS implements GradientOptimizationAlgorithm {

    private Criteria stopCriteria = new Criteria(100, 1e-3);

    private final int m = 10;

    @Override
    public OptimizedResult optimize(List<? extends Variable> latentVariables,
                                    FitnessFunction fitnessFunction,
                                    FitnessFunctionGradient fitnessFunctionGradient) {

        double[] startingPoint = Optimizer.convertToArrayPoint(getAsDoubleTensors(latentVariables));

        DoubleTensor result = maximize(
            new ApacheFitnessFunctionAdapter(fitnessFunction, latentVariables),
            new ApacheFitnessFunctionGradientAdapter(fitnessFunctionGradient, latentVariables),
            DoubleTensor.create(startingPoint)
        );

        Map<VariableReference, DoubleTensor> optimizedValues = Optimizer.convertFromPoint(
            result.asFlatDoubleArray(),
            latentVariables
        );

        return new OptimizedResult(optimizedValues, 0);
    }

    private DoubleTensor maximize(ApacheFitnessFunctionAdapter objFunc,
                                  ApacheFitnessFunctionGradientAdapter objFuncGradient,
                                  DoubleTensor position) {

        final int dimensionCount = Ints.checkedCast(position.getShape()[0]);

        DoubleTensor sVector = DoubleTensor.zeros(m, dimensionCount);
        DoubleTensor yVector = DoubleTensor.zeros(m, dimensionCount);
        final double[] alpha = new double[m];
        final double[] rho = new double[m];

        DoubleTensor gradient;
        DoubleTensor q;
        DoubleTensor gradientPrevious;
        DoubleTensor s;
        DoubleTensor y;

        gradient = DoubleTensor.create(objFuncGradient.value(position.asFlatDoubleArray())).unaryMinusInPlace();
        DoubleTensor positionPrevious = position;

        int iter = 0;
        double H0k = 1.0;

        Criteria current = new Criteria();
        do {

            double relativeEpsilon = 0.0001 * Math.max(1.0, norm(position).scalar());

            if (norm(gradient).scalar() < relativeEpsilon) {
                break;
            }

            //Algorithm 7.4 (L-BFGS two-loop recursion)
            q = gradient;
            final int k = Math.min(m, iter);

            // for i = k − 1, k − 2, . . . , k − m
            for (int i = k - 1; i >= 0; i--) {
                // alpha_i <- rho_i*s_i^T*q

                final DoubleTensor sRow = getRow(sVector, i);
                final DoubleTensor yRow = getRow(yVector, i);

                rho[i] = 1.0 / (dot(sRow, yRow).scalar());
                alpha[i] = rho[i] * dot(sRow, q).scalar();

                // q <- q - alpha_i*y_i
                q = q.minus(yRow.times(alpha[i]));
            }
            // r <- H_k^0*q
            q = q.times(H0k);

            //for i k − m, k − m + 1, . . . , k − 1
            for (int i = 0; i < k; i++) {
                // beta <- rho_i * y_i^T * r

                final DoubleTensor sRow = getRow(sVector, i);
                final DoubleTensor yRow = getRow(yVector, i);

                double beta = rho[i] * dot(yRow, q).scalar();

                // r <- r + s_i * ( alpha_i - beta)
                q = q.plus(sRow.times(alpha[i] - beta));
            }
            // stop with result "H_k*f_f'=q"

            // any issues with the descent direction ?
            double alphaInit = 1.0 / norm(gradient).scalar();

//            double descent = -dot(gradient, q).scalar();
//            if (descent > -0.0001 * relativeEpsilon) {
//                q = gradient.unaryMinus();
//                iter = 0;
//                alphaInit = 1.0;
//            }

            // find step length
            MoreThuente.Results linesearchResult = MoreThuente.linesearch(position, q.unaryMinus(), objFunc, objFuncGradient, alphaInit);


            // update guess
            position = position.minus(q.times(linesearchResult.alpha));
            gradientPrevious = gradient;
            gradient = DoubleTensor.create(objFuncGradient.value(position.asFlatDoubleArray())).unaryMinusInPlace();

            s = position.minus(positionPrevious);
            y = gradient.minus(gradientPrevious);

            // update the history
            if (iter < m) {
                setRow(sVector, iter, s);
                setRow(yVector, iter, y);
            } else {
                sVector = shiftAndAddRow(sVector, s);
                yVector = shiftAndAddRow(yVector, y);
            }

            // update the scaling factor
            double yDoty = dot(y, y).scalar();
            if (Double.isNaN(yDoty)) {
                throw new IllegalStateException();
            }

            if (yDoty == 0.0) {
                throw new IllegalStateException();
            }

            H0k = dot(y, s).divInPlace(yDoty).scalar();

            positionPrevious = position;

            current.gradNorm = gradient.abs().max().scalar();

            iter++;
            current.iterations++;

        } while (!stopCriteria.isConverged(current));

        return position;
    }

    private static DoubleTensor getRow(DoubleTensor matrix, int row) {
        return matrix.slice(0, row);
    }

    private static void setRow(DoubleTensor target, int row, DoubleTensor operand) {
        for (int i = 0; i < target.getShape()[1]; i++) {
            target.setValue(operand.getValue(i), row, i);
        }
    }

    private DoubleTensor shiftAndAddRow(DoubleTensor matrix, DoubleTensor vector) {
        return DoubleTensor.concat(matrix.slice(Slicer.builder().slice(1, m).build()), vector.reshape(1, -1));
    }

    private static DoubleTensor norm(DoubleTensor input) {
        return input.pow(2).sum().sqrtInPlace();
    }

    public static DoubleTensor dot(DoubleTensor left, DoubleTensor right) {
        return left.times(right).sum();
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    private static class Criteria {
        int iterations;
        double gradNorm;

        public enum Status {
            ITERATION_LIMIT, GRAD_NORM_TOLERANCE, CONTINUE
        }

        boolean isConverged(Criteria current) {
            return checkConvergence(current) != Status.CONTINUE;
        }

        public Status checkConvergence(Criteria current) {
            if ((this.iterations > 0) && (current.iterations > this.iterations)) {
                return Status.ITERATION_LIMIT;
            }

            if ((this.gradNorm > 0) && (current.gradNorm < this.gradNorm)) {
                return Status.GRAD_NORM_TOLERANCE;
            }

            return Status.CONTINUE;
        }
    }

}
