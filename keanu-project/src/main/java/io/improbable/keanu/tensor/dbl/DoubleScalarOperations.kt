package io.improbable.keanu.tensor.dbl

import io.improbable.keanu.tensor.FloatingPointScalarOperations
import org.apache.commons.math3.analysis.function.Sigmoid
import org.apache.commons.math3.special.Gamma
import org.apache.commons.math3.util.FastMath

object DoubleScalarOperations : FloatingPointScalarOperations<Double> {

    //Floating point

    private val SIGMOID = Sigmoid()
    private val LOG2 = FastMath.log(2.0)

    override fun sigmoid(value: Double): Double {
        return SIGMOID.value(value)
    }

    override fun reciprocal(value: Double): Double {
        return 1.0 / value
    }

    override fun sqrt(value: Double): Double {
        return FastMath.sqrt(value)
    }

    override fun log(value: Double): Double {
        return FastMath.log(value)
    }

    override fun logGamma(value: Double): Double {
        return Gamma.logGamma(value)
    }

    override fun digamma(value: Double): Double {
        return Gamma.digamma(value)
    }

    override fun trigamma(value: Double): Double {
        return Gamma.trigamma(value)
    }

    override fun sin(value: Double): Double {
        return FastMath.sin(value)
    }

    override fun cos(value: Double): Double {
        return FastMath.cos(value)
    }

    override fun tan(value: Double): Double {
        return FastMath.tan(value)
    }

    override fun atan(value: Double): Double {
        return FastMath.atan(value)
    }

    override fun atan2(x: Double, y: Double): Double {
        return FastMath.atan2(y, x)
    }

    override fun asin(value: Double): Double {
        return FastMath.asin(value)
    }

    override fun acos(value: Double): Double {
        return FastMath.acos(value)
    }

    override fun sinh(value: Double): Double {
        return FastMath.sinh(value)
    }

    override fun cosh(value: Double): Double {
        return FastMath.cosh(value)
    }

    override fun tanh(value: Double): Double {
        return FastMath.tanh(value)
    }

    override fun asinh(value: Double): Double {
        return FastMath.asinh(value)
    }

    override fun acosh(value: Double): Double {
        return FastMath.acosh(value)
    }

    override fun atanh(value: Double): Double {
        return FastMath.atanh(value)
    }

    override fun exp(value: Double): Double {
        return FastMath.exp(value)
    }

    override fun log1p(value: Double): Double {
        return FastMath.log1p(value)
    }

    override fun log2(value: Double): Double {
        return FastMath.log(value) / LOG2
    }

    override fun log10(value: Double): Double {
        return FastMath.log10(value)
    }

    override fun exp2(value: Double): Double {
        return FastMath.pow(2.0, value)
    }

    override fun expM1(value: Double): Double {
        return FastMath.expm1(value)
    }

    override fun ceil(value: Double): Double {
        return FastMath.ceil(value)
    }

    override fun floor(value: Double): Double {
        return FastMath.floor(value)
    }

    override fun round(value: Double): Double {
        if (value >= 0.0) {
            return FastMath.floor(value + 0.5)
        } else {
            return FastMath.ceil(value - 0.5)
        }
    }

    override fun notNan(value: Double): Boolean {
        return !value.isNaN()
    }

    override fun isNaN(value: Double): Boolean {
        return value.isNaN()
    }

    override fun isFinite(value: Double): Boolean {
        return value.isFinite()
    }

    override fun isInfinite(value: Double): Boolean {
        return value.isInfinite()
    }

    override fun isNegativeInfinity(value: Double): Boolean {
        return value == Double.NEGATIVE_INFINITY
    }

    override fun isPositiveInfinity(value: Double): Boolean {
        return value == Double.POSITIVE_INFINITY
    }

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