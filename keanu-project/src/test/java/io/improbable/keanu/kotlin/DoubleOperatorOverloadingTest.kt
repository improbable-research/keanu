package io.improbable.keanu.kotlin

import io.improbable.keanu.DeterministicRule
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex
import junit.framework.TestCase.assertEquals
import org.junit.Rule
import org.junit.Test
import kotlin.math.pow


class DoubleOperatorOverloadingTest {

    @Rule
    @JvmField
    var deterministicRule = DeterministicRule()

    @Test
    fun doubleVertexPlus() {
        val a = GaussianVertex(0.0, 1.0)
        val b = GaussianVertex(0.0, 1.0)

        val e1 = a.value + b.value
        val r1 = a + b
        assertEquals(e1, r1.value)

        val e2 = a.value + 2.0
        val r2 = a + 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 + a.value
        val r3 = 2.0 + a
        assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticDoublePlus() {
        val a = ArithmeticDouble(1.0)
        val b = ArithmeticDouble(1.5)

        val e1 = a.value + b.value
        val r1 = a + b
        assertEquals(e1, r1.value)

        val e2 = a.value + 2.0
        val r2 = a + 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 + a.value
        val r3 = 2.0 + a
        assertEquals(e3, r3.value)
    }

    @Test
    fun doubleVertexMinus() {
        val a = GaussianVertex(0.0, 1.0)
        val b = GaussianVertex(0.0, 1.0)

        val e1 = a.value - b.value
        val r1 = a - b
        assertEquals(e1, r1.value)

        val e2 = a.value - 2.0
        val r2 = a - 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 - a.value
        val r3 = 2.0 - a
        assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticDoubleMinus() {
        val a = ArithmeticDouble(1.0)
        val b = ArithmeticDouble(2.0)

        val e1 = a.value - b.value
        val r1 = a - b
        assertEquals(e1, r1.value)

        val e2 = a.value - 2.0
        val r2 = a - 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 - a.value
        val r3 = 2.0 - a
        assertEquals(e3, r3.value)
    }

    @Test
    fun doubleVertexUnaryMinus() {
        val a = GaussianVertex(0.0, 1.0)

        val e1 = -a.value
        val r1 = -a
        assertEquals(e1, r1.value)
    }

    @Test
    fun arithmeticDoubleUnaryMinus() {
        val a = ArithmeticDouble(1.0)

        val e1 = -a.value
        val r1 = -a
        assertEquals(e1, r1.value)
    }

    @Test
    fun doubleVertexTimes() {
        val a = GaussianVertex(0.0, 1.0)
        val b = GaussianVertex(0.0, 1.0)

        val e1 = a.value * b.value
        val r1 = a * b
        assertEquals(e1, r1.value)

        val e2 = a.value * 2.0
        val r2 = a * 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 * a.value
        val r3 = 2.0 * a
        assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticDoubleTimes() {
        val a = ArithmeticDouble(1.0)
        val b = ArithmeticDouble(2.0)

        val e1 = a.value * b.value
        val r1 = a * b
        assertEquals(e1, r1.value)

        val e2 = a.value * 2.0
        val r2 = a * 2.0
        assertEquals(e2, r2.value)
    }

    @Test
    fun doubleVertexPow() {
        val a = GaussianVertex(0.0, 2.0)
        val b = GaussianVertex(0.0, 3.0)

        val e1 = a.value.pow(b.value)
        val r1 = a.pow(b)
        assertEquals(e1, r1.value)

        val e2 = a.value.pow(3.0)
        val r2 = a.pow(3.0)
        assertEquals(e2, r2.value)

    }

    @Test
    fun arithmeticDoublePow() {
        val a = ArithmeticDouble(2.0)
        val b = ArithmeticDouble(3.0)

        val e1 = a.value.pow(b.value)
        val r1 = a.pow(b)
        assertEquals(e1, r1.value)

        val e2 = a.value.pow(3.0)
        val r2 = a.pow(3.0)
        assertEquals(e2, r2.value)
    }

    @Test
    fun doubleVertexDivide() {
        val a = GaussianVertex(0.0, 1.0)
        val b = ConstantDoubleVertex(2.0)

        val e1 = a.value / b.value
        val r1 = a / b
        assertEquals(e1, r1.value)

        val e2 = a.value / 2.0
        val r2 = a / 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 / a.value
        val r3 = 2.0 / a
        assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticDoubleDivide() {
        val a = ArithmeticDouble(1.0)
        val b = ArithmeticDouble(2.0)

        val e1 = a.value / b.value
        val r1 = a / b
        assertEquals(e1, r1.value)

        val e2 = a.value / 2.0
        val r2 = a / 2.0
        assertEquals(e2, r2.value)

        val e3 = 2.0 / a.value
        val r3 = 2.0 / a
        assertEquals(e3, r3.value)
    }

    @Test
    fun doubleVertexNestedOperators() {
        val a = 4.0
        val b = GaussianVertex(0.0, 1.0)
        val c = GaussianVertex(0.0, 1.0)
        val d = ConstantDoubleVertex(1.0)
        val e = UniformVertex(1.0, 10.0)
        val f = 10.0
        val g = 5.0

        val result = a / ((((b + c - d) / e) * f) - g)
        val expected = a / ((((b.value + c.value - d.value) / e.value) * f) - g)
        assertEquals(expected, result.value)
    }

    @Test
    fun arithmeticDoubleNestedOperators() {
        val a = 4.0
        val b = ArithmeticDouble(1.0)
        val c = ArithmeticDouble(2.0)
        val d = ArithmeticDouble(3.0)
        val e = ArithmeticDouble(10.0)
        val f = 10.0
        val g = 5.0

        val result = a / ((((b + c - d) / e) * f) - g)
        val expected = a / ((((b.value + c.value - d.value) / e.value) * f) - g)
        assertEquals(expected, result.value)
    }

    @Test
    fun doubleVertexPrefixOperators() {
        val a = ConstantDoubleVertex(0.123)

        assertEquals(Math.acos(a.value.scalar()), acos(a).value.scalar())
        assertEquals(Math.asin(a.value.scalar()), asin(a).value.scalar())
        assertEquals(Math.cos(a.value.scalar()), cos(a).value.scalar())
        assertEquals(Math.sin(a.value.scalar()), sin(a).value.scalar())
        assertEquals(Math.exp(a.value.scalar()), exp(a).value.scalar())
        assertEquals(Math.log(a.value.scalar()), log(a).value.scalar())
        assertEquals(Math.pow(a.value.scalar(), 2.345), pow(a, 2.345).value.scalar())
        assertEquals(Math.pow(a.value.scalar(), a.value.scalar()), pow(a, a).value.scalar())
    }

    @Test
    fun arithmeticDoublePrefixOperators() {
        val a = ArithmeticDouble(0.123)

        assertEquals(Math.acos(a.value), acos(a).value)
        assertEquals(Math.asin(a.value), asin(a).value)
        assertEquals(Math.cos(a.value), cos(a).value)
        assertEquals(Math.sin(a.value), sin(a).value)
        assertEquals(Math.exp(a.value), exp(a).value)
        assertEquals(Math.log(a.value), log(a).value)
        assertEquals(Math.pow(a.value, 2.345), pow(a, 2.345).value)
        assertEquals(Math.pow(a.value, a.value), pow(a, a).value)
    }

}
