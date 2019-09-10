package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.linesearch.HagerZhang;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.variational.optimizer.Optimizer.getAsDoubleTensors;

/**
 * https://github.com/miikama/cpp-math/blob/master/CppNumericalSolvers/include/cppoptlib/solver/lbfgssolver.h
 * <p>
 * Pages 177-179 from:
 * http://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf
 */
public class LBFGS implements GradientOptimizationAlgorithm {

    private final Criteria stopCriteria;
    private final int m;
    private final HagerZhang hagerZhang;

    private LBFGS(Criteria stopCriteria, int m, HagerZhang hagerZhang) {
        this.stopCriteria = stopCriteria;
        this.m = m;
        this.hagerZhang = hagerZhang;
    }

    public static LBFGSBuilder builder() {
        return new LBFGSBuilder();
    }

    @Override
    public OptimizedResult optimize(List<? extends Variable> latentVariables,
                                    FitnessFunction fitnessFunction,
                                    FitnessFunctionGradient fitnessFunctionGradient) {

        double[] startingPoint = Optimizer.convertToArrayPoint(getAsDoubleTensors(latentVariables));

        DoubleTensor result = maximize(
            new FitnessFunctionGradientFlatAdapter(fitnessFunctionGradient, latentVariables),
            DoubleTensor.create(startingPoint)
        );

        Map<VariableReference, DoubleTensor> optimizedValues = Optimizer.convertFromPoint(
            result.asFlatDoubleArray(),
            latentVariables
        );

        return new OptimizedResult(optimizedValues, 0);
    }

    private DoubleTensor maximize(FitnessFunctionGradientFlatAdapter objFuncGradient,
                                  DoubleTensor position) {

        ArrayList<DoubleTensor> sQueue = new ArrayList<>();
        ArrayList<DoubleTensor> yQueue = new ArrayList<>();

        final double[] alpha = new double[m];
        final double[] rho = new double[m];

        DoubleTensor gradient;
        DoubleTensor q;
        DoubleTensor r;
        DoubleTensor gradientPrevious;
        DoubleTensor s;
        DoubleTensor y;

        gradient = DoubleTensor.create(objFuncGradient.gradient(position.asFlatDoubleArray())).unaryMinusInPlace();
        DoubleTensor positionPrevious = position;

        int iter = 0;
        double H0k = 1.0;

        Criteria current = new Criteria();
        do {

            //TODO: What's this about?
//            double relativeEpsilon = 0.0001 * Math.max(1.0, norm(position).scalar());
//
//            if (norm(gradient).scalar() < relativeEpsilon) {
//                break;
//            }

            //Algorithm 7.4 (L-BFGS two-loop recursion)
            q = gradient;
            final int k = Math.min(m, iter);

            // for i = k − 1, k − 2, . . . , k − m
            for (int i = k - 1; i >= 0; i--) {
                // alpha_i <- rho_i*s_i^T*q

                final DoubleTensor sRow = sQueue.get(i);
                final DoubleTensor yRow = yQueue.get(i);

                rho[i] = 1.0 / (dot(sRow, yRow).scalar());
                alpha[i] = rho[i] * dot(sRow, q).scalar();

                // q <- q - alpha_i*y_i
                q = q.minus(yRow.times(alpha[i]));
            }

            // r <- H_k^0*q
            r = q.times(H0k);

            //for i k − m, k − m + 1, . . . , k − 1
            for (int i = 0; i < k; i++) {
                // beta <- rho_i * y_i^T * r

                final DoubleTensor sRow = sQueue.get(i);
                final DoubleTensor yRow = yQueue.get(i);

                double beta = rho[i] * dot(yRow, r).scalar();

                // r <- r + s_i * ( alpha_i - beta)
                r = r.plus(sRow.times(alpha[i] - beta));
            }
            // stop with result "H_k*f_f'=q"

            // any issues with the descent direction ?
            double alphaInit = 1.0 / norm(gradient).scalar();

            //TODO: What's this about?
            double descent = -dot(gradient, r).scalar();
            if (descent > 0) {
                r = gradient;
                iter = 0;
                alphaInit = 1.0;
            }

            // find step length
            HagerZhang.Results linesearchResult = hagerZhang.lineSearch(position, r.unaryMinus(), objFuncGradient, alphaInit);

            if (!linesearchResult.isSuccess()) {
                throw new IllegalStateException();
//                return position;
            }

            // update guess
            position = position.minus(r.times(linesearchResult.getAlpha()));
            gradientPrevious = gradient;
            gradient = DoubleTensor.create(objFuncGradient.gradient(position.asFlatDoubleArray())).unaryMinusInPlace();

            s = position.minus(positionPrevious);
            y = gradient.minus(gradientPrevious);

            // update the history
            if (iter < m) {
                yQueue.add(y);
                sQueue.add(s);

            } else {
                yQueue.remove(0);
                yQueue.add(y);

                sQueue.remove(0);
                sQueue.add(s);
            }

            // update the scaling factor
            double yDoty = dot(y, y).scalar();
            if (Double.isNaN(yDoty)) {
                throw new IllegalStateException();
            }

            if (yDoty == 0.0) {
                return position;
            }

            H0k = dot(y, s).divInPlace(yDoty).scalar();

            if (H0k == 0) {
                //TODO: what's this about?
                return position;
            }

            positionPrevious = position;

            current.gradNorm = gradient.abs().max().scalar();

            iter++;
            current.iterations++;

        } while (!stopCriteria.isConverged(current));

        return position;
    }

    private static DoubleTensor norm(DoubleTensor input) {
        return input.pow(2).sum().sqrtInPlace();
    }

    private static DoubleTensor dot(DoubleTensor left, DoubleTensor right) {
        return left.times(right).sum();
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    private static class Criteria {
        int iterations;
        double gradNorm;
        double xTolerance;
        double fTolerance;

        public enum Status {
            ITERATION_LIMIT, GRAD_NORM_TOLERANCE, X_TOLERANCE, F_TOLERANCE, CONTINUE
        }

        boolean isConverged(Criteria current) {
            return checkConvergence(current) != Status.CONTINUE;
        }

        public Status checkConvergence(Criteria current) {
            if (this.iterations > 0 && current.iterations > this.iterations) {
                return Status.ITERATION_LIMIT;
            }

            if (this.gradNorm > 0 && current.gradNorm < this.gradNorm) {
                return Status.GRAD_NORM_TOLERANCE;
            }

            if (this.xTolerance > 0 && current.xTolerance < this.xTolerance) {
                return Status.X_TOLERANCE;
            }

            if (this.fTolerance > 0 && current.fTolerance < this.fTolerance) {
                return Status.F_TOLERANCE;
            }

            return Status.CONTINUE;
        }
    }

    public static class LBFGSBuilder {
        private Criteria stopCriteria = new Criteria(1000, 1e-8, 0, 0);
        private int m = 10;
        private HagerZhang hagerZhang = HagerZhang.builder().build();

        private LBFGSBuilder() {
        }

        public LBFGSBuilder stopCriteria(Criteria stopCriteria) {
            this.stopCriteria = stopCriteria;
            return this;
        }

        public LBFGSBuilder m(int m) {
            this.m = m;
            return this;
        }

        public LBFGSBuilder hagerZhang(HagerZhang hagerZhang) {
            this.hagerZhang = hagerZhang;
            return this;
        }

        public LBFGS build() {
            return new LBFGS(stopCriteria, m, hagerZhang);
        }
    }
}
