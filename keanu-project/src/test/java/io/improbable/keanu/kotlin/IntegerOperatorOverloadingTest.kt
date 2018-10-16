package io.improbable.keanu.kotlin

import io.improbable.keanu.DeterministicRule
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex
import junit.framework.TestCase
import org.junit.Rule
import org.junit.Test

class IntegerOperatorOverloadingTest {

    @Rule
    @JvmField
    var deterministicRule = DeterministicRule()

    @Test
    fun integerVertexPlus() {
        val a = PoissonVertex(1.0)
        val b = PoissonVertex(2.0)

        val e1 = a.value + b.value
        val r1 = a + b
        TestCase.assertEquals(e1, r1.value)

        val e2 = a.value + 2
        val r2 = a + 2
        TestCase.assertEquals(e2, r2.value)

        val e3 = 2 + a.value
        val r3 = 2 + a
        TestCase.assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticIntegerPlus() {
        val a = ArithmeticInteger(1)
        val b = ArithmeticInteger(2)

        val e1 = a.value + b.value
        val r1 = a + b
        TestCase.assertEquals(e1, r1.value)

        val e2 = a.value + 2
        val r2 = a + 2
        TestCase.assertEquals(e2, r2.value)

        val e3 = 2 + a.value
        val r3 = 2 + a
        TestCase.assertEquals(e3, r3.value)
    }

    @Test
    fun integerVertexMinus() {
        val a = PoissonVertex(1.0)
        val b = PoissonVertex(2.0)

        val e1 = a.value - b.value
        val r1 = a - b
        TestCase.assertEquals(e1, r1.value)

        val e2 = a.value - 2
        val r2 = a - 2
        TestCase.assertEquals(e2, r2.value)

        val e3 = 2 - a.value
        val r3 = 2 - a
        TestCase.assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticIntegerMinus() {
        val a = ArithmeticInteger(1)
        val b = ArithmeticInteger(2)

        val e1 = a.value - b.value
        val r1 = a - b
        TestCase.assertEquals(e1, r1.value)

        val e2 = a.value - 2
        val r2 = a - 2
        TestCase.assertEquals(e2, r2.value)

        val e3 = 2 - a.value
        val r3 = 2 - a
        TestCase.assertEquals(e3, r3.value)
    }

    @Test
    fun integerVertexUnaryMinus() {
        val a = PoissonVertex(1.0)

        val e1 = -a.value
        val r1 = -a
        TestCase.assertEquals(e1, r1.value)
    }

    @Test
    fun arithmeticIntegerUnaryMinus() {
        val a = ArithmeticInteger(2)

        val e1 = -a.value
        val r1 = -a
        TestCase.assertEquals(e1, r1.value)
    }

    @Test
    fun integerVertexTimes() {
        val a = PoissonVertex(1.0)
        val b = PoissonVertex(2.0)

        val e1 = a.value * b.value
        val r1 = a * b
        TestCase.assertEquals(e1, r1.value)

        val e2 = a.value * 2
        val r2 = a * 2
        TestCase.assertEquals(e2, r2.value)

        val e3 = 2 * a.value
        val r3 = 2 * a
        TestCase.assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticIntegerTimes() {
        val a = ArithmeticInteger(1)
        val b = ArithmeticInteger(2)

        val e1 = a.value * b.value
        val r1 = a * b
        TestCase.assertEquals(e1, r1.value)

        val e2 = a.value * 2
        val r2 = a * 2
        TestCase.assertEquals(e2, r2.value)

        val e3 = 2 * a.value
        val r3 = 2 * a
        TestCase.assertEquals(e3, r3.value)
    }

    @Test
    fun integerVertexPow() {
        val a = PoissonVertex(2.0)
        val b = PoissonVertex(3.0)

        val e1 = a.value.pow(b.value)
        val r1 = a.pow(b)
        TestCase.assertEquals(e1, r1.value)

        val e2 = a.value.pow(3)
        val r2 = a.pow(3)
        TestCase.assertEquals(e2, r2.value)
    }

    @Test
    fun arithmeticIntegerPow() {
        val a = ArithmeticInteger(2)
        val b = ArithmeticInteger(3)

        val e1 = 8
        val r1 = a.pow(b)
        TestCase.assertEquals(e1, r1.value)

        val e2 = 8
        val r2 = a.pow(3)
        TestCase.assertEquals(e2, r2.value)

        val e3 = 2 * a.value
        val r3 = 2 * a
        TestCase.assertEquals(e3, r3.value)
    }

    @Test
    fun integerVertexDivide() {
        val a = PoissonVertex(1.0)
        val b = ConstantIntegerVertex(2)

        val e1 = a.value / b.value
        val r1 = a / b
        TestCase.assertEquals(e1, r1.value)

        val e2 = a.value / 2
        val r2 = a / 2
        TestCase.assertEquals(e2, r2.value)

        val e3 = 2 / (a.value + 1)
        val r3 = 2 / (a + 1)
        TestCase.assertEquals(e3, r3.value)
    }

    @Test
    fun arithmeticIntegerDivide() {
        val a = ArithmeticInteger(1)
        val b = ArithmeticInteger(2)

        val e1 = a.value / b.value
        val r1 = a / b
        TestCase.assertEquals(e1, r1.value)

        val e2 = a.value / 2
        val r2 = a / 2
        TestCase.assertEquals(e2, r2.value)

        val e3 = 2 / a.value
        val r3 = 2 / a
        TestCase.assertEquals(e3, r3.value)
    }

    @Test
    fun integerVertexNestedOperators() {
        val a = 4
        val b = PoissonVertex(1.0)
        val c = PoissonVertex(2.0)
        val d = ConstantIntegerVertex(1)
        val e = UniformIntVertex(1, 10)
        val f = 10
        val g = 5

        val result = a / ((((b + c - d) / e) * f) - g)
        val expected = a / ((((b.value + c.value - d.value) / e.value) * f) - g)
        TestCase.assertEquals(expected, result.value)
    }

    @Test
    fun arithmeticIntegerNestedOperators() {
        val a = 4
        val b = ArithmeticInteger(1)
        val c = ArithmeticInteger(2)
        val d = ArithmeticInteger(3)
        val e = ArithmeticInteger(10)
        val f = 10
        val g = 5

        val result = a / ((((b + c - d) / e) * f) - g)
        val expected = a / ((((b.value + c.value - d.value) / e.value) * f) - g)
        TestCase.assertEquals(expected, result.value)
    }
}
