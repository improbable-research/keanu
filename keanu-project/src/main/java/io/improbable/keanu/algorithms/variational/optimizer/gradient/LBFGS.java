package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import com.google.common.primitives.Ints;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * https://github.com/miikama/cpp-math/blob/master/CppNumericalSolvers/include/cppoptlib/solver/lbfgssolver.h
 */
public class LBFGS {

    private Criteria stopCriteria = new Criteria(100, 1e-3);

    public void minimize(FitnessFunction objFunc, FitnessFunctionGradient objFuncGradient, DoubleTensor x0) {
        int m = 10;
        int DIM = Ints.checkedCast(x0.getShape()[0]);
//        MatrixType sVector = MatrixType::Zero(DIM, m);
//        MatrixType yVector = MatrixType::Zero(DIM, m);

        DoubleTensor sVector = DoubleTensor.zeros(m, DIM);
        DoubleTensor yVector = DoubleTensor.zeros(m, DIM);

//        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> alpha = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(m);

        DoubleTensor alpha = DoubleTensor.zeros(m);

//        TVector grad(DIM), q(DIM), grad_old(DIM), s(DIM), y(DIM);
        DoubleTensor grad = DoubleTensor.zeros(DIM);
        DoubleTensor q = DoubleTensor.zeros(DIM);
        DoubleTensor grad_old = DoubleTensor.zeros(DIM);
        DoubleTensor s = DoubleTensor.zeros(DIM);
        DoubleTensor y = DoubleTensor.zeros(DIM);

        grad = objFuncGradient.getGradientsAt(x0);
        DoubleTensor x_old = x0;
        DoubleTensor x_old2 = x0;

        int iter = 0, globIter = 0;
        double H0k = 1;
//        this->m_current.reset();

        Criteria current = new Criteria();
        do {
//            const Scalar relativeEpsilon = static_cast<Scalar>(0.0001) * std::max(static_cast<Scalar>(1.0), x0.norm());

            double relativeEpsilon = 0.0001 * Math.max(1.0, norm(x0).scalar());

            if (norm(grad).scalar() < relativeEpsilon)
                break;

            //Algorithm 7.4 (L-BFGS two-loop recursion)
            q = grad;
//            const int k = std::min(m, iter);
            int k = Math.min(m, iter);

            // for i = k − 1, k − 2, . . . , k − m§
            for (int i = k - 1; i >= 0; i--) {
                // alpha_i <- rho_i*s_i^T*q
//                const double rho = 1.0 / static_cast<TVector>(sVector.col(i))
//                    .dot(static_cast<TVector>(yVector.col(i)));

                double rho = 1.0 / (dot(sVector.slice(1, i), yVector.slice(1, i)).scalar());

//                alpha(i) = rho * static_cast<TVector>(sVector.col(i)).dot(q);
                double a = rho * dot(sVector.slice(1, i), q).scalar();
                alpha.setValue(a, i);

                // q <- q - alpha_i*y_i
//                q = q - alpha(i) * yVector.col(i);
                q = q.minus(yVector.slice(1, i).times(a));
            }
            // r <- H_k^0*q
//            q = H0k * q;
            q = q.times(H0k);

            //for i k − m, k − m + 1, . . . , k − 1
            for (int i = 0; i < k; i++) {
                // beta <- rho_i * y_i^T * r

//                const Scalar rho = 1.0 / static_cast < TVector > (sVector.col(i))
//                    .dot(static_cast < TVector > (yVector.col(i)));
                double rho = 1.0 / (dot(sVector.slice(1, i), yVector.slice(1, i)).scalar());

//                const Scalar beta = rho * static_cast < TVector > (yVector.col(i)).dot(q);
                double beta = rho * yVector.slice(1, i).times(q).sum().scalar();

                // r <- r + s_i * ( alpha_i - beta)
//                q = q + sVector.col(i) * (alpha(i) - beta);
                q = q.plus(sVector.slice(1, i).times(alpha.getValue(i) - beta));
            }
            // stop with result "H_k*f_f'=q"

            // any issues with the descent direction ?
            double descent = -grad.times(q).sumNumber();
            double alpha_init = 1.0 / norm(grad).scalar();
            if (descent > -0.0001 * relativeEpsilon) {
//                q = -1 * grad;
                q = grad.times(-1);
                iter = 0;
                alpha_init = 1.0;
            }

            // find steplength
//            double rate = MoreThuente < ProblemType, 1 >::linesearch(x0, -q, objFunc, alpha_init);
            double rate = MoreThuente.linesearch(x0, q.unaryMinus(), objFunc, objFuncGradient, alpha_init);

            // update guess
//            x0 = x0 - rate * q;
            x0 = x0.minus(q.times(rate));

            grad_old = grad;
            grad = objFuncGradient.getGradientsAt(x0);

//            s = x0 - x_old;
            s = x0.minus(x_old);
//            y = grad - grad_old;
            y = grad.minus(grad_old);

            // update the history
            if (iter < m) {
//                sVector.col(iter) = s;
                setRow(sVector, iter, s);
//                yVector.col(iter) = y;
                setRow(yVector, iter, y);
            } else {

//                sVector.leftCols(m - 1) = sVector.rightCols(m - 1).eval();
//                sVector.rightCols(1) = s;
//                yVector.leftCols(m - 1) = yVector.rightCols(m - 1).eval();
//                yVector.rightCols(1) = y;

                shiftAndAddRow(sVector, s);
                shiftAndAddRow(yVector, y);
            }
            // update the scaling factor
//            H0k = y.dot(s) / static_cast < double>(y.dot(y));
            H0k = dot(y, s).divInPlace(dot(y, y)).scalar();

            x_old = x0;
            // std::cout << "iter: "<<globIter<< ", f = " <<  objFunc.value(x0) << ", ||g||_inf "
            // <<gradNorm  << std::endl;

            iter++;
            globIter++;
            current.iterations++;
//            ++this->m_current.iterations;

//            this->m_current.gradNorm = grad.template lpNorm<Eigen::Infinity > ();
            current.gradNorm = grad.abs().max().scalar();

//            this->m_status = checkConvergence(this->m_stop, this->m_current);

//        } while ((objFunc.callback(this->m_current, x0)) &&(this->m_status == Status::Continue));
        } while (!stopCriteria.isConverged(current));

    }

    private static void setRow(DoubleTensor target, int row, DoubleTensor operand) {
        for (int i = 0; i < target.getShape()[0]; i++) {
            target.setValue(operand.getValue(i), row, i);
        }
    }

    private static DoubleTensor shiftAndAddRow(DoubleTensor matrix, DoubleTensor vector){
        return DoubleTensor.concat(matrix.slice(Slicer.builder().slice(1).build()), vector.reshape(1, -1));
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
