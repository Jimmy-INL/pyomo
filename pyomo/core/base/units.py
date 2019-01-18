#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

"""Pyomo Units Module ** Not Yet Complete

This module provides support for including units within Pyomo expressions, and provides
methods for checking the consistency of units within those expresions.

To use this package within your Pyomo model, you first need an instance of a PyomoUnitsContainer.
You can use the module level instance called units_container or you can create your own instance.
With this object you can then use the pre-defined units or even create new units.

Examples:
    To use a unit within an expression, simply reference the unit
    as a field on the units manager.

        >>> from pyomo.environ import *
        >>> from pyomo.units import units_container as uc
        >>> model = ConcreteModel()
        >>> model.acc = Var()
        >>> model.obj = Objective(expr=(model.acc*uc.m/uc.s**2 - 9.81*uc.m/uc.s**2)**2)
        >>>

Notes:
    The implementation is currently based on `pint <http://pint.readthedocs.io>`_
    Python package and supports all the units that are supported by pint.

"""
# """Pyomo Units Module
#
# This module provides support for units checking within Pyomo.
#
# Units can be used within expressions, or passed to :class:`Var <pyomo.environ.Var>` components
# or :class:`Param <pyomo.environ.Param>` components to specify their units.
# To use this package within your Pyomo model, you first need a units manager. There
# is a module level units manager called units_mananger or you can create your own
# instance of the :class:`UnitsManager`. With this object you can then use the pre-defined
# units or create new units.
#
# Examples:
#     To use a unit within an expression, simply reference the unit
#     as a field on the units manager.
#
#         >>> from pyomo.environ import *
#         >>> from pyomo.units import units_manager as um
#         >>> model = ConcreteModel()
#         >>> model.x = Var(units=um.kg)
#         >>> model.obj = Objective(expr=(model.x - 97.2*um.kg)**2)
#
# Notes:
#     The implementation is currently based on `pint <http://pint.readthedocs.io>`_
#     Python package and supports all the units that are supported by pint.
#
# Todo:
#     * create a new pint unit definition file (and load from that file)
#         since the precision in pint is insufficient for 1e-8 constraint tolerances
#     * clean up docstrings (e.g., add Returns to those that need it)
#     * test pickling and un-pickling
#     * Add a convert_value for numerical values only (e.g., for use in initialization and reporting)
#
# """

from pyomo.core.expr.numvalue import NumericValue, nonpyomo_leaf_types, value
from pyomo.core.expr import current as expr
import pint


class UnitsError(Exception):
    """
    An exception base class for all errors/warnings associated with units
    """
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return 'Units Error: ' + str(self.msg)


class InconsistentUnitsError(UnitsError):
    """
    An exception indicating that inconsistent units are present on an expression.

    E.g., x == y, where x is in units of um.kg and y is in units of um.meter
    """
    def __init__(self, exp1, exp2, msg):
        msg = '{} not compatible with {}. {}'.format(str(exp1), str(exp2), str(msg))
        super(InconsistentUnitsError, self).__init__(msg)


class PyomoUnit(NumericValue):
    """An object that represents a single unit in Pyomo (e.g., kg, meter)

    Users do not need to create instances of Unit directly.
    Instead, access units as attributes on an instance of a UnitsManager class.
    """

    def __init__(self, pint_unit, pint_registry):
        self._pint_unit = pint_unit
        self._pint_registry = pint_registry

    def _get_pint_unit(self):
        return self._pint_unit

    def _get_pint_registry(self):
        return self._pint_registry

    # Todo: test pickle and implement __getstate__/__setstate__ to do the right thing

    def getname(self, fully_qualified=False, name_buffer=None):
        return str(self)

    # methods/properties that use the NumericValue base class implementation
    # name property
    # local_name
    # cname

    def is_constant(self):
        """
        Indicates if the NumericValue is constant and can be replaced with a plain old number
        Overloaded from: NumericValue

        This method indicates if the NumericValue is a constant and can be replaced with a plain
        old number. Although units are constant, we do NOT want this replaced - therefore
        we return False here to prevent replacement.

        Returns
        =======
            bool : False (This method always returns False)
        """
        return False

    def is_fixed(self):
        """
        Indicates if the NumericValue is fixed with respect to the "solver".
        Overloaded from: NumericValue

        Indicates if the Unit should be treated as fixed. Since the Unit is always treated as
        a constant value of 1.0, it is always fixed.

        Returns
        =======
            bool : True (This method always returns True)

        """
        # These are fixed values in relation to the solver - they will be set to 1.0 w.r.t. the solver
        return True

    def is_parameter_type(self):
        """ This is not a parameter type (overloaded from NumericValue) """
        return False

    def is_variable_type(self):
        """ This is not a variable type (overloaded from NumericValue) """
        return False

    def is_potentially_variable(self):
        """
        This is not potentially variable (does not and cannot contain a variable).
        Overloaded from NumericValue
        """
        return False

    def is_named_expression_type(self):
        """ This is not a named expression (overloaded from NumericValue) """
        return False

    def is_expression_type(self):
        """ This is a leaf, not an expression (overloaded from NumericValue) """
        return False

    def is_component_type(self):
        """ This is not a component type (overloaded from NumericValue) """
        return False

    def is_relational(self):
        """ This is not relational (overloaded from NumericValue) """
        return False

    def is_indexed(self):
        """ This is not indexed (overloaded from NumericValue) """
        return False

    # polynomial_degree using implementation in NumericValue
    def _compute_polynomial_degree(self, result):
        return 0

    def __float__(self):
        """
        Coerce the value to a floating point

        Raises:
            TypeError
        """
        raise TypeError(
            "Implicit conversion of Pyomo Unit `%s' to a float is "
            "disabled. This error is often the result of treating a unit "
            "as though it were a number (e.g., passing a unit to a built-in "
            "math function). Avoid this error by using Pyomo-provided math "
            "functions."
            % self.name)

    def __int__(self):
        """
        Coerce the value to an integer

        Raises:
            TypeError
        """
        raise TypeError(
            "Implicit conversion of Pyomo Unit `%s' to an int is "
            "disabled. This error is often the result of treating a unit "
            "as though it were a number (e.g., passing a unit to a built-in "
            "math function). Avoid this error by using Pyomo-provided math "
            "functions."
            % self.name)

    # def __lt__(self, other):
    #     """
    #     Less than operator
    #
    #     This method is called when Python processes statements of the form::
    #
    #         self < other
    #         other > self
    #     Raises:
    #         TypeError
    #     """
    #     raise TypeError(
    #         "The Pyomo Unit `%s' cannot exist by itself on one side of "
    #         "an inequality. A unit should be part of a larger expression."
    #         % (self.name))
    #
    # def __gt__(self, other):
    #     """
    #     Greater than operator
    #
    #     This method is called when Python processes statements of the form::
    #
    #         self > other
    #         other < self
    #     Raises:
    #         TypeError
    #     """
    #     raise TypeError(
    #         "The Pyomo Unit `%s' cannot exist by itself on one side of "
    #         "an inequality. A unit should be part of a larger expression."
    #         % (self.name))
    #
    # def __le__(self, other):
    #     """
    #     Less than or equal operator
    #
    #     This method is called when Python processes statements of the form::
    #
    #         self <= other
    #         other >= self
    #     Raises:
    #         TypeError
    #     """
    #     raise TypeError(
    #         "The Pyomo Unit `%s' cannot exist by itself on one side of "
    #         "an inequality. A unit should be part of a larger expression."
    #         % (self.name))
    #
    # def __ge__(self, other):
    #     """
    #     Greater than or equal operator
    #
    #     This method is called when Python processes statements of the form::
    #
    #         self >= other
    #         other <= self
    #     Raises:
    #         TypeError
    #     """
    #     raise TypeError(
    #         "The Pyomo Unit `%s' cannot exist by itself on one side of "
    #         "an inequality. A unit should be part of a larger expression."
    #         % (self.name))
    #
    # def __eq__(self, other):
    #     """
    #     Equal to operator
    #
    #     This method is called when Python processes the statement::
    #
    #         self == other
    #     Raises:
    #         TypeError
    #     """
    #     raise TypeError(
    #         "The Pyomo Unit `%s' cannot exist by itself on one side of "
    #         "a relational operation (e.g., equality). A unit should be "
    #         "part of a larger expression."
    #         % (self.name))
    #
    # def __add__(self, other):
    #     """
    #     Binary addition
    #
    #     This method is called when Python processes the statement::
    #
    #         self + other
    #     Raises:
    #         TypeError
    #     """
    #     raise TypeError(
    #         "The Pyomo Unit `%s' cannot exist by itself on one side of "
    #         "an addition (+). The unit should be part of a larger expression."
    #         % (self.name))
    #
    # def __sub__(self, other):
    #     """
    #     Binary subtraction
    #
    #     This method is called when Python processes the statement::
    #
    #         self - other
    #     Raises:
    #         TypeError
    #     """
    #     raise TypeError(
    #         "The Pyomo Unit `%s' cannot exist by itself on one side of "
    #         "a subtraction (-). The unit should be part of a larger expression."
    #         % (self.name))

    # __mul__ uses NumericValue base class implementation
    # __div__ uses NumericValue base class implementation
    # __truediv__ uses NumericValue base class implementation
    # __pow__ uses NumericValue vase class implementation

    # def __radd__(self, other):
    #     """
    #     Binary addition
    #
    #     This method is called when Python processes the statement::
    #
    #         other + self
    #     Raises:
    #         TypeError
    #     """
    #     raise TypeError(
    #         "The Pyomo Unit `%s' cannot exist by itself on one side of "
    #         "an addition (+). The unit should be part of a larger expression."
    #         % (self.name))
    #
    # def __rsub__(self, other):
    #     """
    #     Binary subtraction
    #
    #     This method is called when Python processes the statement::
    #
    #         other - self
    #     Raises:
    #         TypeError
    #     """
    #     raise TypeError(
    #         "The Pyomo Unit `%s' cannot exist by itself on one side of "
    #         "a subtraction (-). The unit should be part of a larger expression."
    #         % (self.name))

    # __rmul__ uses NumericValue base class implementation
    # __rdiv__ uses NumericValue base class implementation
    # __rtruediv__ uses NumericValue base class implementation

    # def __rpow__(self, other):
    #     """
    #     Binary power
    #
    #     This method is called when Python processes the statement::
    #
    #         other ** self
    #     Raises:
    #         TypeError
    #     """
    #     raise TypeError(
    #         "The Pyomo Unit `%s' cannot exist by itself as an exponent. "
    #         "The unit should be part of a larger expression."
    #         % (self.name))

    def __iadd__(self, other):
        """
        Binary addition

        This method is called when Python processes the statement::

            self += other
        Raises:
            TypeError
        """
        raise TypeError(
            "The Pyomo Unit `%s' is read-only and cannot be the target of an in-place addition (+=)."
            % (self.name))

    def __isub__(self, other):
        """
        Binary subtraction

        This method is called when Python processes the statement::

            self -= other
        Raises:
            TypeError
        """
        raise TypeError(
            "The Pyomo Unit `%s' is read-only and cannot be the target of an in-place subtraction (-=)."
            % (self.name))

    def __imul__(self, other):
        """
        Binary multiplication

        This method is called when Python processes the statement::

            self *= other
        Raises:
            TypeError
        """
        raise TypeError(
            "The Pyomo Unit `%s' is read-only and cannot be the target of an in-place multiplication (*=)."
            % (self.name))

    def __idiv__(self, other):
        """
        Binary division

        This method is called when Python processes the statement::

            self /= other
        Raises:
            TypeError
        """
        raise TypeError(
            "The Pyomo Unit `%s' is read-only and cannot be the target of an in-place division (/=)."
            % (self.name))

    def __itruediv__(self, other):
        """
        Binary division (when __future__.division is in effect)

        This method is called when Python processes the statement::

            self /= other
        Raises:
            TypeError
        """
        raise TypeError(
            "The Pyomo Unit `%s' is read-only and cannot be the target of an in-place division (/=)."
            % (self.name))

    def __ipow__(self, other):
        """
        Binary power

        This method is called when Python processes the statement::

            self **= other
        Raises:
            TypeError
        """
        raise TypeError(
            "The Pyomo Unit `%s' is read-only and cannot be the target of an in-place exponentiation (**=)."
            % (self.name))

    #
    # def __neg__(self):
    #     """
    #     Negation
    #
    #     This method is called when Python processes the statement::
    #
    #         - self
    #     Raises:
    #         TypeError
    #     """
    #     raise TypeError(
    #         "The Pyomo Unit `%s' cannot be used as the target for a negation operator. This "
    #         "can often occur with an expression such as -um.kg*3.0 that should instead be "
    #         " written as -3.0*um.kg or -(um.kg*3.0)."
    #         % (self.name))
    #

    # __pos__ uses NumericValue base class implementation

    # def __abs__(self):
    #     """ Absolute value
    #
    #     This method is called when Python processes the statement::
    #
    #         abs(self)
    #     Raises:
    #         TypeError
    #     """
    #     raise TypeError(
    #         "The Pyomo Unit `%s' cannot exist by itself as an argument to absolute value. "
    #         "The unit should be part of a larger expression."
    #         % (self.name))


    def __str__(self):
        """ Returns a string representing the unit """
        # ToDo: implement the _units_equiv method
        #if _units_equiv(self._pint_unit, self._owner_units_manager._pint_ureg.dimensionless):
        #    return 'dimensionless'

        # I am returning the short form here, could
        # also return str(self._pint_unit) for the long form
        # ToDo: fix this to return the short form of the pint unit
        return '{:~}'.format(self._pint_unit)

    def to_string(self, verbose=None, labeler=None, smap=None,
                  compute_values=False):
        """
        Return a string representation of the expression tree.

        See documentation on NumericValue

        Returns:
            A string representation for the expression tree.
        """
        #ToDo: Expand the functionality here as needed.
        return self.__str__()

    def __nonzero__(self):
        """Unit is treated as a constant value of 1.0. Therefore, it is always nonzero
        Returns: (bool)
        """
        return self.__bool__()

    def __bool__(self):
        """Unit is treated as a constant value of 1.0. Therefore, it is always "True"
        Returns: (bool)"""
        return True

    def __call__(self, exception=True):
        """Unit is treated as a constant value, and this method always returns 1.0"""
        return 1.0

    def pprint(self, ostream=None, verbose=False):
        """Display a user readable string description of this object.
           ToDo:
               * This is not implemented yet. Need to write a method to
                produce a "printable" version of the unit.
        """
        if ostream is None:         #pragma:nocover
            ostream = sys.stdout
        ostream.write(str(self))


class _UnitExtractionVisitor(expr.StreamBasedExpressionVisitor):

    def __init__(self):
        """
        Class used to check and retrieve Pyomo and pint units from expressions
        """
        super(_UnitExtractionVisitor, self).__init__()

    @staticmethod
    def _pint_units_equivalent(lhs, rhs):
        """
        Check if two pint units are equivalent

        Parameters
        ----------
        lhs : pint unit
            first pint unit to compare
        rhs : pint unit
            second pint unit to compare

        Returns
        -------
            bool : True if they are equivalent, and False otherwise
        """
        if lhs == rhs:
            return True

        if (lhs is None and rhs is not None) or \
            (lhs is not None and rhs is None):
            return False

        # Units are not the same actual objects.
        # Convert to pint Quantity objects
        # and use pint to compare
        lhsq = 1.0 * lhs
        rhsq = 1.0 * rhs
        # ToDo: Does this need an approximate "float" check?
        if lhsq == rhsq:
            return True

        return False

    #
    # @staticmethod
    # def return_pint_unit_if_equivalent(list_of_units, pint_registry):
    #     """
    #     This method checks that the units in the list are all equivalent and then
    #     returns the first one in the list
    #
    #     Parameters
    #     ----------
    #     list_of_units : list
    #        a list of pint units (or pint unit expressions)
    #     pint_registry : pint.UnitRegistry
    #        the registry that owns the units - this is used to return dimensionless if the list is empty
    #
    #     Returns
    #     -------
    #         pint.Unit : returns the first pint unit or unit expression in the list (if they are equivalent)
    #     """
    #     if len(list_of_units) == 0:
    #         return pint_registry.dimensionless
    #     elif len(list_of_units) == 1:
    #         return list_of_units[0]
    #
    #     ret_unit = list_of_units[0]
    #     for i in range(1,len(list_of_units)):
    #         if not UnitExtractionVisitor.pint_units_equivalent(ret_unit, list_of_units[i]):
    #             raise InconsistentUnits(ret_unit, list_of_units[i], 'Inconsistent units found in expression'
    #                                     ' (e.g., 3.0*kg + 1.2*m)')
    #
    #     return ret_unit

    def _get_unit_for_equivalent_children(self, node, list_of_unit_tuples):
        """
        Return the unit expressions corresponding to a summation or comparison
        of the child data in list_of_unit_tuples

        Parameters
        ----------
        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
            tuple : (PyomoUnit, pint unit)
        """
        assert len(list_of_unit_tuples) > 0

        # verify that the pint units are equivalent from each
        # of the child nodes - assume that PyomoUnits are equivalent
        pint_unit_0 = list_of_unit_tuples[0][1]
        for i in range(1, len(list_of_unit_tuples)):
            pint_unit_i = list_of_unit_tuples[i][1]
            if not _UnitExtractionVisitor._pint_units_equivalent(pint_unit_0, pint_unit_i):
                raise InconsistentUnitsError(pint_unit_0, pint_unit_i,
                        'Error in units found in expression: {}'.format(str(node)))

        # checks were OK, return the first one in the list
        return (list_of_unit_tuples[0][0], list_of_unit_tuples[0][1])

    def _get_unit_for_product(self, node, list_of_unit_tuples):
        """
        Return the unit expression corresonding to a product
        of the child data in list_of_unit_tuples

        Parameters
        ----------
        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
            tuple : (PyomoUnit, pint unit)
        """
        assert len(list_of_unit_tuples) > 1

        pyomo_unit = None
        pint_unit = None
        for i in range(len(list_of_unit_tuples)):
            pyomo_unit_i = list_of_unit_tuples[i][0]
            pint_unit_i = list_of_unit_tuples[i][1]

            # if pyomo_unit_i and pint_unit_i are None, interpreted as dimensionless
            # but both should be None
            if pyomo_unit_i is None:
                assert pint_unit_i is None
            if pint_unit_i is None:
                assert pyomo_unit_i is None

            if pyomo_unit_i is not None:
                if pyomo_unit is None:
                    assert pint_unit is None
                    pyomo_unit = pyomo_unit_i
                    pint_unit = pint_unit_i
                else:
                    pyomo_unit = pyomo_unit * pyomo_unit_i
                    pint_unit = pint_unit * pint_unit_i

        return (pyomo_unit, pint_unit)

    def _get_unit_for_reciprocal(self, node, list_of_unit_tuples):
        """
        Return the unit expression corresponding to a reciprocal
        of the child data in list_of_unit_tuples

        Parameters
        ----------
        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
            tuple : (PyomoUnit, pint unit)
        """
        assert len(list_of_unit_tuples) == 1

        pyomo_unit = 1.0/list_of_unit_tuples[0][0]
        pint_unit = 1.0/list_of_unit_tuples[0][1]
        return (pyomo_unit, pint_unit)

    def _get_unit_for_pow(self, node, list_of_unit_tuples):
        """
        Return the unit expression corresponding to a pow expression
        with the base and exponent in list_of_unit_tuples

        Parameters
        ----------
        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
            tuple : (PyomoUnit, pint unit)
        """
        assert len(list_of_unit_tuples) == 2

        # this operation is x^y
        # x should be in list_of_unit_tuples[0] and
        # y should be in list_of_unit_tuples[1]
        pyomo_unit_base = list_of_unit_tuples[0][0]
        pint_unit_base = list_of_unit_tuples[0][1]
        pyomo_unit_exponent = list_of_unit_tuples[1][0]
        pint_unit_exponent = list_of_unit_tuples[1][1]

        # check to make sure that the exponent is unitless
        if pyomo_unit_exponent is not None or \
            pint_unit_exponent is not None:
            # ToDo: might have to check for dimensionless here as well
            raise UnitsError("Error in sub-expression: {}. "
                             "Exponents in a pow expression must be unitless."
                             "".format(node))

        # if the base is unitless, return None
        if pyomo_unit_base is None:
            assert pint_unit_base is None
            return (None, None)

        # Since the base is not unitless, make sure that the exponent
        # is a fixed number
        exponent = node.args[1]
        if type(exponent) not in nonpyomo_leaf_types \
            and not (exponent.is_fixed() or exponent.is_constant()):
            raise UnitsError("The base of an exponent has units {}, but "
                             "the exponent is not fixed to a numerical value."
                             "".format(str(list_of_unit_tuples[0][0])))
        exponent_value = value(exponent)
        pyomo_unit = list_of_unit_tuples[0][0]**exponent_value
        pint_unit = list_of_unit_tuples[0][1]**exponent_value

        return (pyomo_unit, pint_unit)

    def _get_unit_for_single_child(self, node, list_of_unit_tuples):
        """
        Return the unit expression corresponding to the (single)
        child in the list_of_unit_tuples

        Parameters
        ----------
        list_of_unit_tuples : list
           This is a list of tuples (one for each of the children) where each tuple
           is a PyomoUnit, pint unit pair

        Returns
        -------
            tuple : (PyomoUnit, pint unit)
        """
        assert len(list_of_unit_tuples) == 1

        pyomo_unit = list_of_unit_tuples[0][0]
        pint_unit = list_of_unit_tuples[0][1]
        return (pyomo_unit, pint_unit)

    node_type_method_map = {
        expr.EqualityExpression: _get_unit_for_equivalent_children,
        expr.InequalityExpression: _get_unit_for_equivalent_children,
        expr.RangedExpression: _get_unit_for_equivalent_children,
        expr.SumExpression: _get_unit_for_equivalent_children,
        expr.NPV_SumExpression: _get_unit_for_equivalent_children,
        expr.ProductExpression: _get_unit_for_product,
        expr.MonomialTermExpression: _get_unit_for_product,
        expr.NPV_ProductExpression: _get_unit_for_product,
        expr.ReciprocalExpression: _get_unit_for_reciprocal,
        expr.NPV_ReciprocalExpression: _get_unit_for_reciprocal,
        expr.PowExpression: _get_unit_for_pow,
        expr.NPV_PowExpression: _get_unit_for_pow,
        expr.NegationExpression: _get_unit_for_single_child,
        expr.NPV_NegationExpression: _get_unit_for_single_child,
        expr.AbsExpression: _get_unit_for_single_child,
        expr.NPV_AbsExpression: _get_unit_for_single_child,
        # ToDo: complete the rest of these
        expr.ExternalFunctionExpression: None,
        expr.GetItemExpression: None,
        expr.Expr_ifExpression: None,
        expr.LinearExpression: None,
        expr.UnaryFunctionExpression: None,
        expr.NPV_ExternalFunctionExpression: None,
        expr.NPV_UnaryFunctionExpression: None,
    }


    def exitNode(self, node, data):
        if expr.is_leaf(node):
            if isinstance(node, PyomoUnit):
                return (node, node._get_pint_unit())

            # ToDo: Check for Var or Param and return their units...

            # I have a leaf, but this is not a PyomoUnit - (treat as dimensionless)
            return (None, None)

        # not a leaf - get the appropriate function for type of the node
        node_type = type(node)
        if node_type in self.node_type_method_map:
            node_func = self.node_type_method_map[node_type]
            if node_func is not None:
                return node_func(self, node, data)
        raise TypeError('An unhandled expression node type: {} was encountered while retrieving the'
                        ' units of an expression'.format(str(node_type)))


def _get_units_tuple(expr):
    pyomo_unit, pint_unit = _UnitExtractionVisitor().walk_expression(expr=expr)
    return (pyomo_unit, pint_unit)


def get_units(expr):
    """
    Return the Pyomo units corresponding to this expression (also performs validation
    and will raise an exception if units are not consistent).

    Parameters
    ----------
    expr : Pyomo expression
        The expression containing the desired units

    Returns
    -------
        Pyomo expression containing only units.
    """
    pyomo_unit, pint_unit = _get_units_tuple(expr=expr)
    # ToDo: If we get dimensionless, should we return None?
    return pyomo_unit


def check_units_consistency(expr, allow_exceptions=True):
    """
    Check the consistency of the units within an expression. IF allow_exceptions is False,
    then this function swallows the exception and returns only True or False. Otherwise,
    it will throw an exception if the units are inconsistent.

    Parameters
    ----------
    expr : Pyomo expression
        The source expression to check.

    allow_exceptions: bool
        True if you want any exceptions to be thrown, False if you only want a boolean
        (and the exception is ignored).

    Returns
    -------
        bool : True if units are consistent, and False if not
    """
    if allow_exceptions is True:
        pyomo_unit, pint_unit = _get_units_tuple(expr=expr)
        # if we make it here, then no exceptions were thrown
        return True

    # allow_exceptions is not True, therefore swallow any exceptions in a try-except
    try:
        pyomo_unit, pint_unit = _get_units_tuple(expr=expr)
    except:
        return False

    return True

# def convert_value(src_value, from_units=None, to_units=None):
#     """
#     This method performs explicit conversion of a numerical value in
#     one unit to a numerical value in another unit.
#     This method returns a new expression object that has the conversion for expr in its units to
#     the units specified by to_units.
#
#     Parameters
#     ----------
#     src_value : float
#        The numeric value that will be converted
#     from_units : Pyomo expression with units
#        The source units for value
#     to_units : Pyomo expression with units
#        The desired target units for the new value
#
#     Returns
#     -------
#        float : The new value (src_value converted from from_units to to_units)
#     """
#     pint_src_unit = self._get_units_from_expression(from_units, pint_version=True)
#     pint_dest_unit = self._get_units_from_expression(to_units, pint_version=True)
#
#     src_quantity = src_value*pint_src_unit
#     dest_quantity = src_quantity.to(pint_dest_unit)
#     return dest_quantity.magnitude


class PyomoUnitsContainer(object):
    """Class that is used to create and contain units in Pyomo.

    This is the class that is used to create, contain, and interact with units in Pyomo.
    The module (:mod:`units`) also contains a module attribute called units_container that is
    a singleton instance of a PyomoUnitsContainer. Once a unit is created in the container,
    the unit can be used in expressions.
    For an overview of the usage of this class, see the module documentation (:mod:`.units`)

    This class is based on the "pint" module. Documentation for available units can be found
    at the following url: https://github.com/hgrecco/pint/blob/master/pint/default_en.txt

    Note: Pre-defined units can be accessed through attributes on the PyomoUnitsContainer
    class; however, these attributes are created dynamically through the __getattr__ method,
    and are not present on the class until they are requested.

    This class also supports creation of new units and even new dimensions. These are advanced
    features that should not be needed often. Check to see if the unit already exists before
    trying to create new units.
    """
    def __init__(self):
        """Create a PyomoUnitsContainer instance. """
        self._pint_ureg = pint.UnitRegistry()

    def _pint_registry(self):
        return self._pint_ureg

    def __getattr__(self, item):
        """
        Here, __getattr__ is implemented to automatically create the necessary unit if
        the attribute does not already exist.

        Parameters
        ----------
        item : str
            the name of the new field requested

        Returns
        -------
            PyomoUnit : returns a PyomoUnit corresponding to the requested attribute,
               or None if it cannot be created.

        """
        # since __getattr__ was called, we must not have this field yet
        # try to build a unit from the requested item
        pint_unit = None
        try:
            pint_unit = getattr(self._pint_ureg, item)
            if pint_unit is not None:
                unit = PyomoUnit(pint_unit, self._pint_ureg)
                setattr(self, item, unit)
                return unit
        except pint.errors.UndefinedUnitError as exc:
            pint_unit = None

        if pint_unit is None:
            raise AttributeError('Attribute {0} not found.'.format(str(item)))

    def create_new_base_dimension(self, dimension_name, base_unit_name):
        """
        Use this method to create a new base dimension (e.g. a new dimension other than Length, Mass) for the unit manager.

        Parameters
        ----------
        dimension_name : str
           name of the new dimension (needs to be unique from other dimension names)

        base_unit_name : str
           base_unit_name: name of the base unit for this dimension

        """
        # ToDo: Error checking - if dimension already exists then we should return a useful error message.
        defn_str = str(base_unit_name) + ' = [' + str(dimension_name) + ']'
        self._pint_ureg.define(defn_str)

    def create_new_unit(self, unit_name, base_unit_name, conv_factor, conv_offset=None):
        """
        Create a new unit that is not already included in the units manager.

        Examples:
            # create a new unit of length called football field that is 1000 yards
            # defines: x (in yards) = y (in football fields) X 100.0
            >>> um.create_new_unit('football_field', 'yards', 100.0)

            # create a new unit of temperature that is half the size of a degree F
            # defines x (in K) = y (in half degF) X 10/9 + 255.3722222 K
            >>> um.create_new_unit('half_degF', 'kelvin', 10.0/9.0, 255.3722222)

        Parameters
        ----------
        unit_name : str
           name of the new unit to create
        base_unit_name : str
           name of the base unit from the same "dimension" as the new unit
        conv_factor : float
           value of the multiplicative factor needed to convert the new unit
           to the base unit
        conv_offset : float
           value of any offset between the new unit and the base unit
           Note that the units of this offset are the same as the base unit,
           and it is applied after the factor conversion
           (e.g., base_value = new_value*conv_factor + conv_offset)

        """
        if conv_offset is None:
            defn_str = '{0!s} = {1:g} * {2!s}'.format(unit_name, float(conv_factor), base_unit_name)
        else:
            defn_str = '{0!s} = {1:17.16g} * {2!s}; offset: {3:17.16g}'.format(unit_name, float(conv_factor), base_unit_name,
                                                                     float(conv_offset))
        self._pint_ureg.define(defn_str)



units_container = PyomoUnitsContainer()

