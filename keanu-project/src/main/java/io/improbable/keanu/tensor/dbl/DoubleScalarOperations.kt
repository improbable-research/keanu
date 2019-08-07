package io.improbable.keanu.tensor.dbl

import io.improbable.keanu.tensor.FloatingPointScalarOperations
import org.apache.commons.math3.util.FastMath

object DoubleScalarOperations : FloatingPointScalarOperations<Double> {

    //Floating point

    override fun safeLogTimes(left: Double, right: Double): Double {
        if (right == 0.0) {
            return 0.0
        } else {
            return FastMath.log(left) * right
        }
    }

    override fun logAddExp(left: Double, right: Double): Double {
        val max = Math.max(left, right)
        return max + FastMath.log(FastMath.exp(left - max) + FastMath.exp(right - max))
    }

    private val LOG2 = FastMath.log(2.0)

    override fun logAddExp2(left: Double, right: Double): Double {
        val max = Math.max(left, right);
        return max + FastMath.log(FastMath.pow(2.0, left - max) + FastMath.pow(2.0, right - max)) / LOG2
    }

    // Number Ops

    override fun sub(left: Double, right: Double): Double {
        return left - right
    }

    override fun add(left: Double, right: Double): Double {
        return left + right
    }

    override fun rsub(left: Double, right: Double): Double {
        return right - left
    }

    override fun mul(left: Double, right: Double): Double {
        return left * right
    }

    override fun div(left: Double, right: Double): Double {
        return left / right
    }

    override fun rdiv(left: Double, right: Double): Double {
        return right / left
    }

    override fun pow(left: Double, right: Double): Double {
        return FastMath.pow(left, right)
    }

    override fun abs(value: Double): Double {
        return kotlin.math.abs(value)
    }

    override fun sign(value: Double): Double {
        return kotlin.math.sign(value)
    }

    override fun unaryMinus(value: Double): Double {
        return -value
    }

    override fun max(left: Double, right: Double): Double {
        return kotlin.math.max(left, right)
    }

    override fun min(left: Double, right: Double): Double {
        return kotlin.math.min(left, right)
    }

    //Comparisons

    override fun equalToMask(left: Double, right: Double): Double {
        return if (left == right) 1.0 else 0.0
    }

    override fun gtMask(left: Double, right: Double): Double {
        return if (left > right) 1.0 else 0.0
    }

    override fun gteMask(left: Double, right: Double): Double {
        return if (left >= right) 1.0 else 0.0
    }

    override fun ltMask(left: Double, right: Double): Double {
        return if (left < right) 1.0 else 0.0
    }

    override fun lteMask(left: Double, right: Double): Double {
        return if (left <= right) 1.0 else 0.0
    }

    override fun equalTo(left: Double, right: Double): Boolean {
        return left.equals(right)
    }

    override fun equalToWithinEpsilon(left: Double, right: Double, epsilon: Double): Boolean {
        return kotlin.math.abs(left - right) <= epsilon
    }

    override fun gt(left: Double, right: Double): Boolean {
        return left > right
    }

    override fun gte(left: Double, right: Double): Boolean {
        return left >= right
    }

    override fun lt(left: Double, right: Double): Boolean {
        return left < right
    }

    override fun lte(left: Double, right: Double): Boolean {
        return left <= right
    }

}