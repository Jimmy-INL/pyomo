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
from pyomo.environ import *
from pyomo.core.base.template_expr import IndexTemplate
from pyomo.core.expr import inequality
import pyomo.core.expr.current as expr
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

        self.assertTrue(check_units_consistency(kg < m.kg, uc))
        self.assertTrue(check_units_consistency(kg > m.kg, uc))
        self.assertTrue(check_units_consistency(kg <= m.kg, uc))
        self.assertTrue(check_units_consistency(kg >= m.kg, uc))
        self.assertTrue(check_units_consistency(kg == m.kg, uc))
        self.assertTrue(check_units_consistency(kg + m.kg, uc))
        self.assertTrue(check_units_consistency(kg - m.kg, uc))

        with self.assertRaises(InconsistentUnitsError):
            check_units_consistency(kg + 3, uc)

        with self.assertRaises(InconsistentUnitsError):
            check_units_consistency(kg - 3, uc)

        with self.assertRaises(InconsistentUnitsError):
            check_units_consistency(3 + kg, uc)

        with self.assertRaises(InconsistentUnitsError):
            check_units_consistency(3 - kg, uc)

        # should not assert
        # check __mul__
        self.assertTrue(str(get_units(kg*3, uc)), 'kg')
        # check __rmul__
        self.assertTrue(str(get_units(3*kg, uc)), 'kg')
        # check div / truediv
        self.assertTrue(str(get_units(kg/3.0, uc)), 'kg')
        # check rdiv / rtruediv
        self.assertTrue(str(get_units(3.0/kg, uc)), '(1/kg)')
        # check pow
        self.assertTrue(str(get_units(kg**2, uc)), 'kg**2')

        # check rpow
        x = 2 ** kg  # creation is allowed, only fails when units are "checked"
        self.assertFalse(check_units_consistency(x, uc, allow_exceptions=False))
        with self.assertRaises(UnitsError):
            check_units_consistency(x, uc)

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

        self.assertTrue(str(get_units(-kg, uc)), 'kg')
        self.assertTrue(str(get_units(+kg, uc)), 'kg')
        self.assertTrue(str(get_units(abs(kg), uc)), 'kg')

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

    def _get_check_units_ok(self, x, pyomo_units_container, str_check=None, expected_type=None):
        if expected_type is not None:
            self.assertEqual(expected_type, type(x))

        self.assertTrue(check_units_consistency(x, pyomo_units_container))
        if str_check is not None:
            self.assertEqual(str_check, str(get_units(x, pyomo_units_container)))
        else:
            self.assertIsNone(get_units(x, pyomo_units_container))

    def _get_check_units_fail(self, x, pyomo_units_container, expected_type=None, expected_error=InconsistentUnitsError):
        if expected_type is not None:
            self.assertEqual(expected_type, type(x))

        self.assertFalse(check_units_consistency(x, pyomo_units_container, allow_exceptions=False))
        with self.assertRaises(expected_error):
            check_units_consistency(x, pyomo_units_container, allow_exceptions=True)

        with self.assertRaises(expected_error):
            # allow_exceptions=True should also be the default
            check_units_consistency(x, pyomo_units_container)

        # we also expect get_units to fail
        with self.assertRaises(expected_error):
            get_units(x, pyomo_units_container)

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
        self._get_check_units_ok(3.0*kg == 1.0*kg, uc, 'kg', expr.EqualityExpression)
        self._get_check_units_fail(3.0*kg == 2.0*m, uc, expr.EqualityExpression)

        # test inequality
        self._get_check_units_ok(3.0*kg <= 1.0*kg, uc, 'kg', expr.InequalityExpression)
        self._get_check_units_fail(3.0*kg <= 2.0*m, uc, expr.InequalityExpression)
        self._get_check_units_ok(3.0*kg >= 1.0*kg, uc, 'kg', expr.InequalityExpression)
        self._get_check_units_fail(3.0*kg >= 2.0*m, uc, expr.InequalityExpression)

        # test RangedExpression
        self._get_check_units_ok(inequality(3.0*kg, 4.0*kg, 5.0*kg), uc, 'kg', expr.RangedExpression)
        self._get_check_units_fail(inequality(3.0*m, 4.0*kg, 5.0*kg), uc, expr.RangedExpression)
        self._get_check_units_fail(inequality(3.0*kg, 4.0*m, 5.0*kg), uc, expr.RangedExpression)
        self._get_check_units_fail(inequality(3.0*kg, 4.0*kg, 5.0*m), uc, expr.RangedExpression)

        # test SumExpression, NPV_SumExpression
        self._get_check_units_ok(3.0*model.x*kg + 1.0*model.y*kg + 3.65*model.z*kg, uc, 'kg', expr.SumExpression)
        self._get_check_units_fail(3.0*model.x*kg + 1.0*model.y*m + 3.65*model.z*kg, uc, expr.SumExpression)

        self._get_check_units_ok(3.0*kg + 1.0*kg + 2.0*kg, uc, 'kg', expr.NPV_SumExpression)
        self._get_check_units_fail(3.0*kg + 1.0*kg + 2.0*m, uc, expr.NPV_SumExpression)

        # test ProductExpression, NPV_ProductExpression
        self._get_check_units_ok(model.x*kg * model.y*m, uc, 'kg*m', expr.ProductExpression)
        self._get_check_units_ok(3.0*kg * 1.0*m, uc, 'kg*m', expr.NPV_ProductExpression)
        self._get_check_units_ok(3.0*kg*m, uc, 'kg*m', expr.NPV_ProductExpression)
        # I don't think that there are combinations that can "fail" for products

        # test MonomialTermExpression
        self._get_check_units_ok(model.x*kg, uc, 'kg', expr.MonomialTermExpression)

        # test ReciprocalExpression, NPV_ReciprocalExpression
        self._get_check_units_ok(1.0/(model.x*kg), uc, '(1/kg)', expr.ReciprocalExpression)
        self._get_check_units_ok(1.0/kg, uc, '(1/kg)', expr.NPV_ReciprocalExpression)
        # I don't think that there are combinations that can "fail" for products

        # test PowExpression, NPV_PowExpression
        # ToDo: fix the str representation to combine the powers or the expression system
        self._get_check_units_ok((model.x*kg**2)**3, uc, 'kg**2**3', expr.PowExpression) # would want this to be kg**6
        self._get_check_units_fail(kg**model.x, uc, expr.PowExpression, UnitsError)
        self._get_check_units_fail(model.x**kg, uc, expr.PowExpression, UnitsError)
        self._get_check_units_ok(kg**2, uc, 'kg**2', expr.NPV_PowExpression)
        self._get_check_units_fail(3.0**kg, uc, expr.NPV_PowExpression, UnitsError)

        # test NegationExpression, NPV_NegationExpression
        self._get_check_units_ok(-(kg*model.x*model.y), uc, 'kg', expr.NegationExpression)
        self._get_check_units_ok(-kg, uc, 'kg', expr.NPV_NegationExpression)
        # don't think there are combinations that fan "fail" for negation

        # test AbsExpression, NPV_AbsExpression
        self._get_check_units_ok(abs(kg*model.x), uc, 'kg', expr.AbsExpression)
        self._get_check_units_ok(abs(kg), uc, 'kg', expr.NPV_AbsExpression)
        # don't think there are combinations that fan "fail" for abs

        # test the different UnaryFunctionExpression / NPV_UnaryFunctionExpression types
        # log
        self._get_check_units_ok(log(3.0*model.x), uc, None, expr.UnaryFunctionExpression)
        self._get_check_units_fail(log(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(log(3.0*model.p), uc, None, expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(log(3.0*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # log10
        self._get_check_units_ok(log10(3.0*model.x), uc, None, expr.UnaryFunctionExpression)
        self._get_check_units_fail(log10(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(log10(3.0*model.p), uc, None, expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(log10(3.0*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # sin
        self._get_check_units_ok(sin(3.0*model.x*uc.radians), uc, None, expr.UnaryFunctionExpression)
        self._get_check_units_fail(sin(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_fail(sin(3.0*kg*model.x*uc.kg), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(sin(3.0*model.p*uc.radians), uc, None, expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(sin(3.0*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # cos
        self._get_check_units_ok(cos(3.0*model.x*uc.radians), uc, None, expr.UnaryFunctionExpression)
        self._get_check_units_fail(cos(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_fail(cos(3.0*kg*model.x*uc.kg), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(cos(3.0*model.p*uc.radians), uc, None, expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(cos(3.0*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # tan
        self._get_check_units_ok(tan(3.0*model.x*uc.radians), uc, None, expr.UnaryFunctionExpression)
        self._get_check_units_fail(tan(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_fail(tan(3.0*kg*model.x*uc.kg), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(tan(3.0*model.p*uc.radians), uc, None, expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(tan(3.0*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # sin
        self._get_check_units_ok(sinh(3.0*model.x*uc.radians), uc, None, expr.UnaryFunctionExpression)
        self._get_check_units_fail(sinh(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_fail(sinh(3.0*kg*model.x*uc.kg), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(sinh(3.0*model.p*uc.radians), uc, None, expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(sinh(3.0*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # cos
        self._get_check_units_ok(cosh(3.0*model.x*uc.radians), uc, None, expr.UnaryFunctionExpression)
        self._get_check_units_fail(cosh(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_fail(cosh(3.0*kg*model.x*uc.kg), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(cosh(3.0*model.p*uc.radians), uc, None, expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(cosh(3.0*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # tan
        self._get_check_units_ok(tanh(3.0*model.x*uc.radians), uc, None, expr.UnaryFunctionExpression)
        self._get_check_units_fail(tanh(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_fail(tanh(3.0*kg*model.x*uc.kg), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(tanh(3.0*model.p*uc.radians), uc, None, expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(tanh(3.0*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # asin
        self._get_check_units_ok(asin(3.0*model.x), uc, 'rad', expr.UnaryFunctionExpression)
        self._get_check_units_fail(asin(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(asin(3.0*model.p), uc, 'rad', expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(asin(3.0*model.p*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # acos
        self._get_check_units_ok(acos(3.0*model.x), uc, 'rad', expr.UnaryFunctionExpression)
        self._get_check_units_fail(acos(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(acos(3.0*model.p), uc, 'rad', expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(acos(3.0*model.p*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # atan
        self._get_check_units_ok(atan(3.0*model.x), uc, 'rad', expr.UnaryFunctionExpression)
        self._get_check_units_fail(atan(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(atan(3.0*model.p), uc, 'rad', expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(atan(3.0*model.p*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # exp
        self._get_check_units_ok(exp(3.0*model.x), uc, None, expr.UnaryFunctionExpression)
        self._get_check_units_fail(exp(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(exp(3.0*model.p), uc, None, expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(exp(3.0*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # sqrt
        self._get_check_units_ok(sqrt(3.0*model.x), uc, None, expr.UnaryFunctionExpression)
        self._get_check_units_ok(sqrt(3.0*model.x*kg**2), uc, 'sqrt(kg**2)', expr.UnaryFunctionExpression)
        self._get_check_units_ok(sqrt(3.0*model.x*kg), uc, 'sqrt(kg)', expr.UnaryFunctionExpression)
        self._get_check_units_ok(sqrt(3.0*model.p), uc, None, expr.NPV_UnaryFunctionExpression)
        self._get_check_units_ok(sqrt(3.0*model.p*kg**2), uc, 'sqrt(kg**2)', expr.NPV_UnaryFunctionExpression)
        self._get_check_units_ok(sqrt(3.0*model.p*kg), uc, 'sqrt(kg)', expr.NPV_UnaryFunctionExpression)
        # asinh
        self._get_check_units_ok(asinh(3.0*model.x), uc, 'rad', expr.UnaryFunctionExpression)
        self._get_check_units_fail(asinh(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(asinh(3.0*model.p), uc, 'rad', expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(asinh(3.0*model.p*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # acosh
        self._get_check_units_ok(acosh(3.0*model.x), uc, 'rad', expr.UnaryFunctionExpression)
        self._get_check_units_fail(acosh(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(acosh(3.0*model.p), uc, 'rad', expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(acosh(3.0*model.p*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # atanh
        self._get_check_units_ok(atanh(3.0*model.x), uc, 'rad', expr.UnaryFunctionExpression)
        self._get_check_units_fail(atanh(3.0*kg*model.x), uc, expr.UnaryFunctionExpression, UnitsError)
        self._get_check_units_ok(atanh(3.0*model.p), uc, 'rad', expr.NPV_UnaryFunctionExpression)
        self._get_check_units_fail(atanh(3.0*model.p*kg), uc, expr.NPV_UnaryFunctionExpression, UnitsError)
        # ceil
        self._get_check_units_ok(ceil(kg*model.x), uc, 'kg', expr.UnaryFunctionExpression)
        self._get_check_units_ok(ceil(kg), uc, 'kg', expr.NPV_UnaryFunctionExpression)
        # don't think there are combinations that fan "fail" for ceil
        # floor
        self._get_check_units_ok(floor(kg*model.x), uc, 'kg', expr.UnaryFunctionExpression)
        self._get_check_units_ok(floor(kg), uc, 'kg', expr.NPV_UnaryFunctionExpression)
        # don't think there are combinations that fan "fail" for floor

        # test Expr_ifExpression
        # consistent if, consistent then/else
        self._get_check_units_ok(expr.Expr_if(IF=model.x*kg + kg >= 2.0*kg, THEN=model.x*kg, ELSE=model.y*kg),
                                 uc, 'kg', expr.Expr_ifExpression)
        # unitless if, consistent then/else
        self._get_check_units_ok(expr.Expr_if(IF=model.x >= 2.0, THEN=model.x*kg, ELSE=model.y*kg),
                                 uc, 'kg', expr.Expr_ifExpression)
        # consistent if, unitless then/else
        self._get_check_units_ok(expr.Expr_if(IF=model.x*kg + kg >= 2.0*kg, THEN=model.x, ELSE=model.x),
                                 uc, None, expr.Expr_ifExpression)
        # inconsistent then/else
        self._get_check_units_fail(expr.Expr_if(IF=model.x >= 2.0, THEN=model.x*m, ELSE=model.y*kg),
                                 uc, expr.Expr_ifExpression)
        # inconsistent then/else NPV
        self._get_check_units_fail(expr.Expr_if(IF=model.x >= 2.0, THEN=model.p*m, ELSE=model.p*kg),
                                 uc, expr.Expr_ifExpression)
        # inconsistent then/else NPV units only
        self._get_check_units_fail(expr.Expr_if(IF=model.x >= 2.0, THEN=m, ELSE=kg),
                                 uc, expr.Expr_ifExpression)

        # test IndexTemplate and GetItemExpression
        model.S = Set()
        i = IndexTemplate(model.S)
        j = IndexTemplate(model.S)
        self._get_check_units_ok(i, uc, None, IndexTemplate)

        model.vv = Var(model.S, model.S)
        self._get_check_units_ok(model.vv[i,j+1], uc, None, expr.GetItemExpression)

        # ToDo: complete the remaining expressions
        # test ExternalFunctionExpression, NPV_ExternalFunctionExpression

        # test LinearExpression

    def test_unit_creation(self):
        uc = PyomoUnitsContainer()

        uc.create_new_unit('football_field', 'yards', 100.0)

        ff = uc.football_field
        yds = uc.yards
        self.assertTrue(check_units_consistency(3.0*ff - 2.0*ff, uc))
        # we do NOT support implicit conversion
        self.assertFalse(check_units_consistency(3.0*ff - 2.0*yds, uc, allow_exceptions=False))

        valid_expr = 3.0*ff * 100.0 * yds/ff == 300.0 * yds
        self.assertTrue(check_units_consistency(valid_expr, uc))

        # ToDo: test convert functionality once complete

    def test_dimension_creation(self):
        pass


if __name__ == "__main__":
    unittest.main()
