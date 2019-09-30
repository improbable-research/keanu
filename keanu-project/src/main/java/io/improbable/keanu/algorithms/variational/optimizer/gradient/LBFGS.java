package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.linesearch.HagerZhang;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.Value;

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
    private final int correctionCount;
    private final HagerZhang hagerZhang;

    private LBFGS(Criteria stopCriteria, int correctionCount, HagerZhang hagerZhang) {
        this.stopCriteria = stopCriteria;
        this.correctionCount = correctionCount;
        this.hagerZhang = hagerZhang;
    }

    public static LBFGSBuilder builder() {
        return new LBFGSBuilder();
    }

    @Override
    public OptimizedResult optimize(List<? extends Variable> latentVariables,
                                    FitnessFunctionGradient fitnessFunctionGradient) {

        double[] startingPoint = Optimizer.convertToArrayPoint(getAsDoubleTensors(latentVariables));

        FitnessFunctionGradientFlatAdapter fitnessAndGradientAdaptor = new FitnessFunctionGradientFlatAdapter(
            fitnessFunctionGradient,
            latentVariables
        );

        PositionAndFitness result = maximize(
            fitnessAndGradientAdaptor,
            DoubleTensor.create(startingPoint)
        );

        Map<VariableReference, DoubleTensor> optimizedValues = Optimizer.convertFromPoint(
            result.position.asFlatDoubleArray(),
            latentVariables
        );

        return new OptimizedResult(optimizedValues, result.fitness);
    }

    @Value
    @AllArgsConstructor
    private static class PositionAndFitness {
        final double fitness;
        final DoubleTensor position;
    }

    private PositionAndFitness maximize(FitnessFunctionGradientFlatAdapter objFuncGradient,
                                        DoubleTensor position) {

        ArrayList<DoubleTensor> sQueue = new ArrayList<>();
        ArrayList<DoubleTensor> yQueue = new ArrayList<>();

        final double[] alpha = new double[correctionCount];
        final double[] rho = new double[correctionCount];

        DoubleTensor q;
        DoubleTensor r;
        DoubleTensor gradientPrevious;
        DoubleTensor s;
        DoubleTensor y;

        FitnessAndGradientFlat fitnessAndGradient = objFuncGradient.fitnessAndGradient(position.asFlatDoubleArray());
        DoubleTensor gradient = DoubleTensor.create(fitnessAndGradient.getGradient()).unaryMinusInPlace();

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
            final int k = Math.min(correctionCount, iter);

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
                return new PositionAndFitness(fitnessAndGradient.getFitness(), position);
            }

            // update guess
            position = position.minus(r.times(linesearchResult.getAlpha()));
            gradientPrevious = gradient;
            fitnessAndGradient = objFuncGradient.fitnessAndGradient(position.asFlatDoubleArray());
            gradient = DoubleTensor.create(fitnessAndGradient.getGradient()).unaryMinusInPlace();

            s = position.minus(positionPrevious);
            y = gradient.minus(gradientPrevious);

            // update the history
            if (iter < correctionCount) {
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
                return new PositionAndFitness(fitnessAndGradient.getFitness(), position);
            }

            H0k = dot(y, s).divInPlace(yDoty).scalar();

            if (H0k == 0) {
                //TODO: what's this about?
                return new PositionAndFitness(fitnessAndGradient.getFitness(), position);
            }

            positionPrevious = position;

            current.gradNorm = gradient.abs().max().scalar();

            iter++;
            current.iterations++;

        } while (!stopCriteria.isConverged(current));

        return new PositionAndFitness(fitnessAndGradient.getFitness(), position);
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
        private Criteria stopCriteria = new Criteria(1000, 1e-2, 0, 0);
        private int correctionCount = 10;
        private HagerZhang hagerZhang = HagerZhang.builder().build();

        private LBFGSBuilder() {
        }

        public LBFGSBuilder stopCriteria(Criteria stopCriteria) {
            this.stopCriteria = stopCriteria;
            return this;
        }

        public LBFGSBuilder correctionCount(int correctionCount) {
            this.correctionCount = correctionCount;
            return this;
        }

        public LBFGSBuilder hagerZhang(HagerZhang hagerZhang) {
            this.hagerZhang = hagerZhang;
            return this;
        }

        public LBFGS build() {
            return new LBFGS(stopCriteria, correctionCount, hagerZhang);
        }
    }
}
