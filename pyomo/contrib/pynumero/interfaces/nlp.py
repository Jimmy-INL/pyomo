#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""
The pyomo.contrib.pynumero.interfaces.nlp module includes classes that
represent nonlinear programmig problems. The NLPs considered in this module 
have the following form:

minimize             f(x)
subject to         c(x) = 0
              d_L <= d(x) <= d_U
              x_L <=  x   <= x_U

where x \in R^{n_x} are the primal variables,
      x_L \in R^{n_x} are the lower bounds of the primal variables,
      x_U \in R^{n_x} are the uppper bounds of the primal variables,
      c: R^{n_x} \rightarrow R^{n_c} are the equality constraints
      d: R^{n_x} \rightarrow R^{n_d} are the inequality constraints
      
.. rubric:: Contents

"""
import pyomo
import pyomo.environ as aml
import six
import abc

__all__ = ['NLP']

#TODO: Decide if we need the condensing matrix stuff
# we could simplify this interface and provide some utility functions
# to create the necessary masks if an algorithm needed it
@six.add_metaclass(abc.ABCMeta)
class NLP(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractproperty
    def n_primals(self):
        """
        Returns number of primal variables
        """
        pass

    @abc.abstractproperty
    def n_eq_constraints(self):
        """
        Returns number of equality constraints
        """
        pass
    
    @abc.abstractproperty
    def n_ineq_constraints(self):
        """
        Returns number of inequality constraints
        """
        pass
    
    @abc.abstractproperty
    def nnz_jacobian_eq(self):
        """
        Returns number of nonzero values in jacobian of equality constraints
        """
        pass

    @abc.abstractproperty
    def nnz_jacobian_ineq(self):
        """
        Returns number of nonzero values in jacobian of inequality constraints
        """
        pass

    @abc.abstractproperty
    def nnz_hessian_lag(self):
        """
        Returns number of nonzero values in hessian of the lagrangian function
        """
        pass

    @abc.abstractmethod
    def primals_lb(self, condensed=False):
        """
        Returns vector of lower bounds for the primal variables

        Parameters
        ----------
        condensed: bool, optional
            True if vector excludes -numpy.inf values (default False)

        Returns
        -------
        vector-like

        """
        pass

    @abc.abstractmethod
    def primals_ub(self, condensed=False):
        """
        Returns vector of upper bounds for the primal variables

        Parameters
        ----------
        condensed: bool, optional
            True if vector excludes numpy.inf values (default False)

        Returns
        -------
        vector-like

        """
        pass

    @abc.abstractmethod
    def ineq_lb(self, condensed=False):
        """
        Returns vector of lower bounds for inequality constraints

        Parameters
        ----------
        condensed: bool, optional
            True if vector excludes -numpy.inf values (default False)

        Returns
        -------
        vector-like

        """
        pass

    @abc.abstractmethod
    def ineq_ub(self, condensed=False):
        """
        Returns vector of upper bounds for inequality constraints

        Parameters
        ----------
        condensed: bool, optional
            True if vector excludes numpy.inf values (default False)

        Returns
        -------
        vector-like

        """
        pass

    @abc.abstractmethod
    def init_primals(self):
        """
        Returns vector with initial values for the primal variables
        """
        pass

    @abc.abstractmethod
    def init_duals_eq(self):
        """
        Returns vector with initial values for the dual variables of the
        equality constraints
        """
        pass

    @abc.abstractmethod
    def init_duals_ineq(self):
        """
        Returns vector with initial values for the dual variables of the
        inequality constraints
        """
        pass

    @abc.abstractmethod
    def create_new_vector(self, vector_type):
        """
        Creates a vector of the appropriate length and structure as 
        requested

        Parameters
        ----------
        vector_type: {'primals', 'primals_lb_condensed', 'primals_ub_condensed', 'eq', 'ineq', 'duals_eq', 'duals_ineq', 'ineq_lb_condensed', 'ineq_ub_condensed'}
            String identifying the appropriate  vector  to create.

        Returns
        -------
        vector-like
        """
        pass

    @abc.abstractmethod
    def set_primals(self, primals):
        """Set the value of the primal variables to be used
        in calls to the evaluation methods

        Parameters
        ----------
        primals: vector_like
            Vector with the values of primal variables.
        """
        pass

    @abc.abstractmethod
    def set_duals_eq(self, duals_eq):
        """Set the value of the dual variables for the equality constraints
        to be used in calls to the evaluation methods (hessian_lag)

        Parameters
        ----------
        duals_eq: vector_like
            Vector with the values of dual variables for the equality constraints
        """
        pass

    @abc.abstractmethod
    def set_duals_ineq(self, duals_ineq):
        """Set the value of the dual variables for the inequality constraints
        to be used in calls to the evaluation methods (hessian_lag)

        Parameters
        ----------
        duals_ineq: vector_like
            Vector with the values of dual variables for the inequality constraints
        """
        pass


    @abc.abstractmethod
    def evaluate_objective(self):
        """Returns value of objective function evaluated at the 
        values given for the primal variables in set_primals

        Returns
        -------
        float
        """
        pass

    @abc.abstractmethod
    def evaluate_grad_objective(self, out=None):
        """Returns gradient of the objective function evaluated at the 
        values given for the primal variables in set_primals

        Parameters
        ----------
        out: vector_like, optional
            Output vector. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        vector_like
        """
        pass

    @abc.abstractmethod
    def evaluate_eq_constraints(self, out=None):
        """Returns the values for the equality constraints evaluated at
        the values given for the primal variales in set_primals

        Parameters
        ----------
        out: array_like, optional
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        vector_like
        """
        pass

    @abc.abstractmethod
    def evaluate_ineq_constraints(self, out=None):
        """Returns the values of the inequality constraints evaluated at
        the values given for the primal variables in set_primals

        Parameters
        ----------
        out : array_like, optional
            Output array. Its type is preserved and it
            must be of the right shape to hold the output.

        Returns
        -------
        vector_like
        """
        pass

    @abc.abstractmethod
    def evaluate_jacobian_eq(self, out=None):
        """Returns the Jacobian of the equality constraints evaluated
        at the values given for the primal variables in set_primals

        Parameters
        ----------
        out : matrix_like (e.g., coo_matrix), optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        matrix_like
        """
        pass

    @abc.abstractmethod
    def evaluate_jacobian_ineq(self, out=None):
        """Returns the Jacobian of the inequality constraints evaluated
        at the values given for the primal variables in set_primals

        Parameters
        ----------
        out : matrix_like (e.g., coo_matrix), optional
            Output matrix with the structure of the jacobian already defined.

        Returns
        -------
        matrix_like
        """
        pass

    @abc.abstractmethod
    def evaluate_hessian_lag(self, out=None):
        """Return the Hessian of the Lagrangian function evaluated
        at the values given for the primal variables in set_primals and
        the dual variables in set_duals_eq and set_duals_ineq

        Parameters
        ----------
        out : matrix_like (e.g., coo_matrix), optional
            Output matrix with the structure of the hessian already defined. Optional

        Returns
        -------
        matrix_like
        """
        pass

    @abc.abstractmethod
    def primals_lb_condensing_matrix(self):
        """Returns a matrix object that projects the full vector of primal variables
        to the condensed lb vector (i.e. vector that excludes the non-finite bounds)."""
        pass

    @abc.abstractmethod
    def primals_ub_condensing_matrix(self):
        """Returns a matrix object that projects the full vector of primal variables
        to the condensed ub vector (i.e. vector that excludes the non-finite bounds)."""
        pass

    @abc.abstractmethod
    def ineq_lb_condensing_matrix(self):
        """Returns a matrix object that projects the full vector of inequality constraints
        to the condensed lb vector (i.e. vector that excludes the non-finite bounds)."""
        pass

    @abc.abstractmethod
    def ineq_ub_condensing_matrix(self):
        """Returns a matrix object that projects the full vector of inequality constraints
        to the condensed vector (i.e. vector that excludes the non-finite bounds)."""
        pass

    @abc.abstractmethod
    def report_solver_status(self, status_code, status_message, primals, duals_eq, duals_ineq):
        """Report the solver status to the appropriate class"""
        pass
