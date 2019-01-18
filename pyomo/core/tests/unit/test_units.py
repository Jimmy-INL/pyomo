#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
#

import pyutilib.th as unittest
from pyomo.core.expr import inequality
import pyomo.core.expr.current as expr
from pyomo.environ import ConcreteModel, Var, value, Param, exp
from pyomo.core.base.units import PyomoUnitsContainer, get_units, InconsistentUnitsError, \
    UnitsError, check_units_consistency, _get_units_tuple

class TestPyomoUnit(unittest.TestCase):

    def test_PyomoUnit_NumericValueMethods(self):
        m = ConcreteModel()
        uc = PyomoUnitsContainer()
        kg = uc.kg

        self.assertEqual(kg.getname(), 'kg')
        self.assertEqual(kg.name, 'kg')
        self.assertEqual(kg.local_name, 'kg')

        m.kg = uc.kg

        self.assertEqual(m.kg.name, 'kg')
        self.assertEqual(m.kg.local_name, 'kg')

        self.assertEqual(kg.is_constant(), False)
        self.assertEqual(kg.is_fixed(), True)
        self.assertEqual(kg.is_parameter_type(), False)
        self.assertEqual(kg.is_variable_type(), False)
        self.assertEqual(kg.is_potentially_variable(), False)
        self.assertEqual(kg.is_named_expression_type(), False)
        self.assertEqual(kg.is_expression_type(), False)
        self.assertEqual(kg.is_component_type(), False)
        self.assertEqual(kg.is_relational(), False)
        self.assertEqual(kg.is_indexed(), False)
        self.assertEqual(kg._compute_polynomial_degree(None), 0)

        with self.assertRaises(TypeError):
            x = float(kg)
        with self.assertRaises(TypeError):
            x = int(kg)

        self.assertTrue(check_units_consistency(kg < m.kg))
        self.assertTrue(check_units_consistency(kg > m.kg))
        self.assertTrue(check_units_consistency(kg <= m.kg))
        self.assertTrue(check_units_consistency(kg >= m.kg))
        self.assertTrue(check_units_consistency(kg == m.kg))
        self.assertTrue(check_units_consistency(kg + m.kg))
        self.assertTrue(check_units_consistency(kg - m.kg))

        with self.assertRaises(InconsistentUnitsError):
            check_units_consistency(kg + 3)

        with self.assertRaises(InconsistentUnitsError):
            check_units_consistency(kg - 3)

        with self.assertRaises(InconsistentUnitsError):
            check_units_consistency(3 + kg)

        with self.assertRaises(InconsistentUnitsError):
            check_units_consistency(3 - kg)

        # should not assert
        # check __mul__
        self.assertTrue(str(get_units(kg*3)), 'kg')
        # check __rmul__
        self.assertTrue(str(get_units(3*kg)), 'kg')
        # check div / truediv
        self.assertTrue(str(get_units(kg/3.0)), 'kg')
        # check rdiv / rtruediv
        self.assertTrue(str(get_units(3.0/kg)), '(1/kg)')
        # check pow
        self.assertTrue(str(get_units(kg**2)), 'kg**2')

        # check rpow
        x = 2 ** kg  # creation is allowed, only fails when units are "checked"
        self.assertFalse(check_units_consistency(x, allow_exceptions=False))
        with self.assertRaises(UnitsError):
            check_units_consistency(x)

        with self.assertRaises(TypeError):
            x = kg
            x += 3

        with self.assertRaises(TypeError):
            x = kg
            x -= 3

        with self.assertRaises(TypeError):
            x = kg
            x *= 3

        with self.assertRaises(TypeError):
            x = kg
            x **= 3

        self.assertTrue(str(get_units(-kg)), 'kg')
        self.assertTrue(str(get_units(+kg)), 'kg')
        self.assertTrue(str(get_units(abs(kg))), 'kg')


        self.assertEqual(str(kg), 'kg')
        self.assertEqual(kg.to_string(), 'kg')
        # ToDo: is this really the correct behavior for verbose?
        self.assertEqual(kg.to_string(verbose=True), 'kg')
        self.assertEqual(kg.to_string(), 'kg')
        self.assertEqual(kg.to_string(), 'kg')

        # check __nonzero__ / __bool__
        self.assertEqual(bool(kg), True)

        # __call__ returns 1.0
        self.assertEqual(kg(), 1.0)
        self.assertEqual(value(kg), 1.0)

    def _get_check_units_ok(self, x, str_check=None, expected_type=None):
        if expected_type is not None:
            self.assertEqual(type(x), expected_type)

        self.assertTrue(check_units_consistency(x))
        if str_check is not None:
            self.assertEqual(str(get_units(x)), str_check)
        else:
            self.assertIsNone(get_units(x))

    def _get_check_units_fail(self, x, expected_type=None, expected_error=InconsistentUnitsError):
        if expected_type is not None:
            self.assertEqual(type(x), expected_type)

        self.assertFalse(check_units_consistency(x, allow_exceptions=False))
        with self.assertRaises(expected_error):
            check_units_consistency(x, allow_exceptions=True)

        with self.assertRaises(expected_error):
            # allow_exceptions=True should also be the default
            check_units_consistency(x)

        # we also expect get_units to fail
        with self.assertRaises(expected_error):
            get_units(x)

    def test_get_check_units_on_all_expressions(self):
        # this method is going to test all the expression types that should work
        # to be defensive, we will also test that we actually have the expected expression type
        # therefore, if the expression system changes and we get a different expression type,
        # we will know we need to change these tests

        uc = PyomoUnitsContainer()
        kg = uc.kg
        m = uc.m

        model = ConcreteModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.p = Param(initialize=42.0, mutable=True)

        # test equality
        self._get_check_units_ok(3.0*kg == 1.0*kg, 'kg', expr.EqualityExpression)
        self._get_check_units_fail(3.0*kg == 2.0*m, expr.EqualityExpression)

        # test inequality
        self._get_check_units_ok(3.0*kg <= 1.0*kg, 'kg', expr.InequalityExpression)
        self._get_check_units_fail(3.0*kg <= 2.0*m, expr.InequalityExpression)
        self._get_check_units_ok(3.0*kg >= 1.0*kg, 'kg', expr.InequalityExpression)
        self._get_check_units_fail(3.0*kg >= 2.0*m, expr.InequalityExpression)

        # test RangedExpression
        self._get_check_units_ok(inequality(3.0*kg, 4.0*kg, 5.0*kg), 'kg', expr.RangedExpression)
        self._get_check_units_fail(inequality(3.0*m, 4.0*kg, 5.0*kg), expr.RangedExpression)
        self._get_check_units_fail(inequality(3.0*kg, 4.0*m, 5.0*kg), expr.RangedExpression)
        self._get_check_units_fail(inequality(3.0*kg, 4.0*kg, 5.0*m), expr.RangedExpression)

        # test SumExpression, NPV_SumExpression
        self._get_check_units_ok(3.0*model.x*kg + 1.0*model.y*kg + 3.65*model.z*kg, 'kg', expr.SumExpression)
        self._get_check_units_fail(3.0*model.x*kg + 1.0*model.y*m + 3.65*model.z*kg, expr.SumExpression)

        self._get_check_units_ok(3.0*kg + 1.0*kg + 2.0*kg, 'kg', expr.NPV_SumExpression)
        self._get_check_units_fail(3.0*kg + 1.0*kg + 2.0*m, expr.NPV_SumExpression)

        # test ProductExpression, NPV_ProductExpression
        self._get_check_units_ok(model.x*kg * model.y*m, 'kg*m', expr.ProductExpression)
        self._get_check_units_ok(3.0*kg * 1.0*m, 'kg*m', expr.NPV_ProductExpression)
        self._get_check_units_ok(3.0*kg*m, 'kg*m', expr.NPV_ProductExpression)
        # I don't think that there are combinations that can "fail" for products

        # test MonomialTermExpression
        self._get_check_units_ok(model.x*kg, 'kg', expr.MonomialTermExpression)

        # test ReciprocalExpression, NPV_ReciprocalExpression
        self._get_check_units_ok(1.0/(model.x*kg), '(1/kg)', expr.ReciprocalExpression)
        self._get_check_units_ok(1.0/kg, '(1/kg)', expr.NPV_ReciprocalExpression)
        # I don't think that there are combinations that can "fail" for products

        # test PowExpression, NPV_PowExpression
        # ToDo: fix the str representation to combine the powers or the expression system
        self._get_check_units_ok((model.x*kg**2)**3, 'kg**2**3', expr.PowExpression) # would want this to be kg**6
        self._get_check_units_fail(kg**model.x, expr.PowExpression, UnitsError)
        self._get_check_units_fail(model.x**kg, expr.PowExpression, UnitsError)
        self._get_check_units_ok(kg**2, 'kg**2', expr.NPV_PowExpression)
        self._get_check_units_fail(3.0**kg, expr.NPV_PowExpression, UnitsError)

        # test NegationExpression, NPV_NegationExpression
        self._get_check_units_ok(-(kg*model.x*model.y), 'kg', expr.NegationExpression)
        self._get_check_units_ok(-kg, 'kg', expr.NPV_NegationExpression)
        # don't think there are combinations that fan "fail" for negation

        # test AbsExpression, NPV_AbsExpression
        self._get_check_units_ok(abs(kg*model.x), 'kg', expr.AbsExpression)
        self._get_check_units_ok(abs(kg), 'kg', expr.NPV_AbsExpression)
        # don't think there are combinations that fan "fail" for abs

        # test UnaryFunctionExpression, NPV_UnaryFunctionExpression
        self._get_check_units_ok(exp(3.0*model.x), None, expr.UnaryFunctionExpression)
        self._get_check_units_fail(exp(3.0*kg*model.x), expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(exp(3.0*model.p), None, expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(exp(3.0*kg), expr.NPV_UnaryFunctionExpression, UnitsError)

        # ToDo: complete the remaining expressions

    def test_unit_creation(self):
        uc = PyomoUnitsContainer()

        uc.create_new_unit('football_field', 'yards', 100.0)

        ff = uc.football_field
        yds = uc.yards
        self.assertTrue(check_units_consistency(3.0*ff - 2.0*ff))
        # we do NOT support implicit conversion
        self.assertFalse(check_units_consistency(3.0*ff - 2.0*yds, allow_exceptions=False))

        valid_expr = 3.0*ff * 100.0 * yds/ff == 300.0 * yds
        self.assertTrue(check_units_consistency(valid_expr))

        # ToDo: test convert functionality once complete

    def test_dimension_creation(self):
        pass


if __name__ == "__main__":
    unittest.main()
