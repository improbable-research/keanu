package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

// based on:
// L-BFGS-B: A LIMITED MEMORY ALGORITHM FOR BOUND CONSTRAINED OPTIMIZATION
// Richard H. Byrd, Peihuang Lu, Jorge Nocedal and Ciyou Zhu
public class LBFGSB {

    /*MatrixType*/ DoubleTensor W, M;
    /*Scalar*/ double theta;
    int DIM;
    int m_historySize = 5;


    @Data
    @AllArgsConstructor
    public static class Pair<A, B> {
        A a;
        B b;
    }

    @Data
    public static class Problem {
        //Vectors
        double[] upperBound;
        double[] lowerBound;
    }

    /**
     * Sort pairs (k,v) according v ascending
     *
     * @param v [description]
     * @return [description]
     */
    private List<Integer> sortIndexes(List<Pair<Integer, Double>> v) {
        return v.stream()
            .sorted(Comparator.comparing(Pair::getB))
            .map(Pair::getA)
            .collect(Collectors.toList());
    }

    //, TVector x_cauchy, VariableTVector c
    DoubleTensor x_cauchy;
    DoubleTensor c;

    /**
     * @brief Algorithm CP: Computation of the generalized Cauchy point
     * @details PAGE 8
     */
    void getGeneralizedCauchyPoint(Problem problem, DoubleTensor x, DoubleTensor g) {

        double epsilon = 1e-8;

        int DIM = (int) x.getLength();

        double[] upperBound = problem.getUpperBound();
        double[] lowerBound = problem.getLowerBound();

        // Given x,l,u,g, and B = \theta I-WMW
        // {all t_i} = { (idx,value), ... }
        // TODO: use "std::set" ?
        List<Pair<Integer, Double>> SetOfT = new ArrayList<>();

        // the feasible set is implicitly given by "SetOfT - {t_i==0}"
        DoubleTensor d = g.times(-1.0);

        // n operations
        double[] dFlat = d.asFlatDoubleArray();
        double[] gFlat = g.asFlatDoubleArray();
        double[] xFlat = x.asFlatDoubleArray();

        for (int j = 0; j < DIM; j++) {

            if (gFlat[j] == 0) {

                //SetOfT.push_back(std::make_pair (j, std::numeric_limits < Scalar >::max()));
                SetOfT.add(new Pair<>(j, Double.MAX_VALUE));
            } else {

                double tmp = 0;

                if (gFlat[j] < 0) {
                    tmp = (xFlat[j] - upperBound[j]) / gFlat[j];
                } else {
                    tmp = (xFlat[j] - lowerBound[j]) / gFlat[j];
                }

                //SetOfT.push_back(std::make_pair (j, tmp));
                SetOfT.add(new Pair<>(j, tmp));

                if (tmp == 0) {
                    dFlat[j] = 0;
                }
            }
        }

        // sortedindices [1,0,2] means the minimal element is on the 1-st entry
        //std::vector < int>sortedIndices = sort_indexes(SetOfT);
        List<Integer> sortedIndices = sortIndexes(SetOfT);

        x_cauchy = x;

        // Initialize
        // p :=     W^Scalar*p
        // (2mn operations)
        d = DoubleTensor.create(dFlat, 1, dFlat.length);

        DoubleTensor p = (W.transpose().matrixMultiply(d));

        // c := 0
        //VariableTVector::Zero (W.cols());
        c = DoubleTensor.zeros(1, W.getShape()[1]);

        // f' :=    g^Scalar*d = -d^Td
        // (n operations)
        double f_prime = g.transpose().matrixMultiply(d).scalar();

        // f'' :=   \theta*d^Scalar*d-d^Scalar*W*M*W^Scalar*d = -\theta*f' - p^Scalar*M*p
        double f_doubleprime = (-theta) * f_prime - p.transpose().matrixMultiply(M).matrixMultiply(p).scalar(); // (O(m^2) operations)

        //f_doubleprime = std::max < Scalar > (std::numeric_limits < Scalar >::epsilon (), f_doubleprime);

        f_doubleprime = Math.max(f_doubleprime, epsilon);

        double f_dp_orig = f_doubleprime;

        // \delta t_min :=  -f'/f''
        double dt_min = -f_prime / f_doubleprime;

        // t_old := 0
        double t_old = 0;

        // b :=     argmin {t_i , t_i >0}
        int i = 0;
        for (int j = 0; j < DIM; j++) {
            i = j;
            if (SetOfT.get(sortedIndices.get(j)).b > 0) {
                break;
            }
        }

        int b = sortedIndices.get(i);

        // see below
        // t :=  min{t_i : i in F}
        double t = SetOfT.get(b).b;

        // \delta Scalar :=  t - 0
        double dt = t;

        double[] x_cauchy_flat = x_cauchy.asFlatDoubleArray();

        // examination of subsequent segments
        while ((dt_min >= dt) && (i < DIM)) {

            if (dFlat[b] > 0) {
                x_cauchy_flat[b] = upperBound[b];
            } else if (dFlat[b] < 0) {
                x_cauchy_flat[b] = lowerBound[b];
            }

            // z_b = x_p^{cp} - x_b
            double zb = x_cauchy_flat[b] - xFlat[b];

            // c   :=  c +\delta t*p
            c = c.plus(p.times(dt));

            // cache
            // VariableTVector wbt = W.row(b);
            DoubleTensor wbt = W.slice(0, b).reshape(1, DIM);

            // f_prime += dt * f_doubleprime + (Scalar) g(b) * g(b) + (Scalar) theta * g(b) * zb - (Scalar) g(b) * wbt.transpose() * (M * c);

            f_prime = f_prime +
                dt * f_doubleprime +
                gFlat[b] * gFlat[b] +
                theta * gFlat[b] * zb - gFlat[b] * wbt.transpose().matrixMultiply(M.matrixMultiply(c)).scalar();

            //f_doubleprime += (Scalar) - 1.0 * theta * g(b) * g(b)
            //                       - (Scalar) 2.0 * (g(b) * (wbt.dot(M * p)))
            //                       - (Scalar) g(b) * g(b) * wbt.transpose() * (M * wbt);

            f_doubleprime = f_doubleprime -
                theta * gFlat[b] * gFlat[b] -
                2.0 * gFlat[b] * (wbt.matrixMultiply(M.matrixMultiply(p))).scalar() -
                gFlat[b] * gFlat[b] * wbt.transpose().matrixMultiply(M.matrixMultiply(wbt)).scalar();

            //f_doubleprime = std::max < Scalar > (std::numeric_limits < Scalar >::epsilon () * f_dp_orig, f_doubleprime);
            f_doubleprime = Math.max(epsilon * f_dp_orig, f_doubleprime);

            p = p.plus(wbt.transpose().times(gFlat[b]));

            dFlat[b] = 0;

            dt_min = -f_prime / f_doubleprime;

            t_old = t;

            ++i;

            if (i < DIM) {
                b = sortedIndices.get(i);
                t = SetOfT.get(b).b;
                dt = t - t_old;
            }
        }

        //dt_min = std::max < Scalar > (dt_min, (Scalar) 0.0);
        dt_min = Math.max(dt_min, 0.0);

        t_old += dt_min;

        for (int ii = i; ii < x_cauchy.getShape()[0]; ii++) {

            x_cauchy_flat[sortedIndices.get(ii)] = xFlat[sortedIndices.get(ii)] + t_old * dFlat[sortedIndices.get(ii)];
        }

        //c += dt_min * p;
        c = c.plus(p.times(dt_min));

        x_cauchy = DoubleTensor.create(x_cauchy_flat);
    }

    /**
     * @param FreeVariables [description]
     * @return find alpha* = max {a : a <= 1 and  l_i-xc_i <= a*d_i <= u_i-xc_i}
     */
    double findAlpha(Problem problem, double[] x_cp, double[] du, int[] FreeVariables) {
        double alphastar = 1;
        final int n = FreeVariables.length;

        //assert(du.rows() == n);

        double[] upperBound = problem.getUpperBound();
        double[] lowerBound = problem.getLowerBound();

        for (int i = 0; i < n; i++) {
            if (du[i] > 0) {
                alphastar = Math.min(alphastar, (upperBound[FreeVariables[i]] - x_cp[FreeVariables[i]]) / du[i]);
            } else {
                alphastar = Math.min(alphastar, (lowerBound[FreeVariables[i]] - x_cp[FreeVariables[i]]) / du[i]);
            }
        }
        return alphastar;
    }

}
