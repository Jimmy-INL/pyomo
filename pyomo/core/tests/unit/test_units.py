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
from pyomo.environ import *
from pyomo.core.base.units import PyomoUnitsContainer, get_units, InconsistentUnitsError

class TestPyomoUnit(unittest.TestCase):

    def test_NumericValueMethods(self):
        m = ConcreteModel()
        uc = PyomoUnitsContainer()
        kg = uc.kg

        self.assertEqual(kg.getname(), 'kg')
        self.assertEqual(kg.name, 'kg')
        self.assertEqual(kg.local_name, 'kg')

        uc = PyomoUnitsContainer()
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

        with self.assertRaises(TypeError):
            x = kg < m.kg

        with self.assertRaises(TypeError):
            x = kg > m.kg

        with self.assertRaises(TypeError):
            x = kg <= m.kg

        with self.assertRaises(TypeError):
            x = kg >= m.kg

        with self.assertRaises(TypeError):
            x = kg == m.kg

        with self.assertRaises(TypeError):
            x = kg + m.kg

        with self.assertRaises(TypeError):
            x = kg - m.kg

        with self.assertRaises(TypeError):
            x = kg + 3

        with self.assertRaises(TypeError):
            x = kg - 3

        with self.assertRaises(TypeError):
            x = 3 + kg

        with self.assertRaises(TypeError):
            x = 3 - kg

        # should not assert
        # check __mul__
        x = kg*3
        # check __rmul__
        x = 3*kg
        # check div / truediv
        x = kg/3.0
        # check rdiv / rtruediv
        x = 3.0/kg
        # check pow
        x = kg**2

        # check rpow
        with self.assertRaises(TypeError):
            x = 2**kg

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

        with self.assertRaises(TypeError):
            x = -kg

        # should not assert
        x = +kg

        with self.assertRaises(TypeError):
            x = abs(kg)

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

    def _check_get_units_ok_kg(self, x, uc):
        x_units = get_units(x, skip_expensive_checks=False, pyomo_units_container=uc)
        self.assertEqual(str(x_units), 'kg')
        x_units = get_units(x, skip_expensive_checks=True, pyomo_units_container=uc)
        self.assertEqual(str(x_units), 'kg')

    def _check_get_units_inconsistent(self, x, uc):
        with self.assertRaises(InconsistentUnitsError):
            x_units = get_units(x, skip_expensive_checks=False, pyomo_units_container=uc)
        with self.assertRaises(InconsistentUnitsError):
            x_units = get_units(x, skip_expensive_checks=True, pyomo_units_container=uc)

    def test_expressions(self):
        uc = PyomoUnitsContainer()
        kg = uc.kg
        m = uc.m

        model = ConcreteModel()
        model.v = Var()



        # test equality
        self._check_get_units_ok_kg(1.0*kg == 4.0*kg, uc)
        self._check_get_units_inconsistent(3.0*kg == 4.0*m, uc)

        # test inequality
        self._check_get_units_ok_kg(3.0*kg <= 4.0*kg, uc)
        self._check_get_units_inconsistent(3.0*kg <= 4.0*m, uc)

        # test RangedExpression
        self._check_get_units_ok_kg(inequality(3.0*kg, 4.0*kg, 5.0*kg, uc), uc)
        self._check_get_units_inconsistent(inequality(3.0*m, 4.0*kg, 5.0*kg), uc)
        self._check_get_units_inconsistent(inequality(3.0*kg, 4.0*m, 5.0*kg), uc)
        self._check_get_units_inconsistent(inequality(3.0*kg, 4.0*kg, 5.0*m), uc)

        # test SumExpression
        self._check_get_units_ok_kg(3.0*model.v*kg + 1.0*kg + 3.65*kg, uc)
        self._check_get_units_inconsistent(3.0*model.v*kg + 1.0*model.v*m, uc)
        # "long" sums should not raise with skip_expensive_checks is True, even if error exists
        x_units = get_units(1.0*kg + 2.0*model.v*kg + 3.0*kg + 4.0*kg, skip_expensive_checks=True, pyomo_units_container=uc)
        x_units = get_units(1.0*kg + 2.0*m + 3.0*model.v*kg + 4.0*kg, skip_expensive_checks=True, pyomo_units_container=uc)

        # test NPV_SumExpression
        self._check_get_units_ok_kg(3.0*kg + 1.0*kg + 3.65*kg, uc)
        self._check_get_units_inconsistent(3.0*kg + 1.0*m, uc)
        # "long" sums should not raise with skip_expensive_checks is True, even if error exists
        x_units = get_units(1.0*kg + 2.0*kg + 3.0*kg + 4.0*kg, skip_expensive_checks=True, pyomo_units_container=uc)
        x_units = get_units(1.0*kg + 2.0*m + 3.0*kg + 4.0*kg, skip_expensive_checks=True, pyomo_units_container=uc)

        #



if __name__ == "__main__":
    unittest.main()
