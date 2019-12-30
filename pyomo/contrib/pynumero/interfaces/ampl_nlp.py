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
This module defines the classes that provide an NLP interface based on
the Ampl Solver Library (ASL) implementation
"""
import pyomo
import pyomo.environ as aml

try:
    import pyomo.contrib.pynumero.extensions.asl as _asl
except ImportError as e:
    print('{}'.format(e))
    raise ImportError('Error importing asl while running nlp interface. '
                      'Make sure libpynumero_ASL is installed and added to path.')

from scipy.sparse import coo_matrix, csr_matrix
import abc
import numpy as np
import tempfile
import os
import six
import shutil
from pyomo.contrib.pynumero.interfaces.nlp import NLP

__all__ = ['AmplNLP']


# ToDo: need to add support for modifying bounds.
# modification of bounds requires rebuiding the maps.
# support for variable bounds seems trivial.
# support for constraint bounds would require more work. (this is not frequent tho)
# TODO: currently, length of upper and lower bounds are all the same (only maps g to ineq)
# TODO: currently, this code always cache's evaluation of gradients and objectives. This may
# result in unnecessary copies
class AslNLP(NLP):
    def __init__(self, nl_file):
        """
        Base class for NLP classes based on the Ampl Solver Library and 
        NL files.

        Parameters
        ----------
        nl_file : string
            filename of the NL-file containing the model
        """
        super(AslNLP, self).__init__()

        # nl file
        self._nl_file = nl_file

        # initialize the ampl interface
        self._asl = _asl.AmplInterface(self._nl_file)

        # collect the NLP structure and key data
        self._collect_nlp_structure()

        # create vectors to store the values for the primals and the duals
        # TODO: Check if we should initialize these to zero or from the init values
        self._primals = self._init_primals.copy()
        self._duals_eq = self._init_duals_eq.copy()
        self._duals_ineq = self._init_duals_ineq.copy()
        self._duals_g = self._init_duals_g.copy()
        self._cached_objective = None
        self._cached_grad_objective = self.create_new_vector('primals')
        self._cached_g = np.zeros(self._n_g)
        self._cached_eq = self.create_new_vector('eq')
        self._cached_ineq = self.create_new_vector('ineq')
        self._cached_jacobian_g_data = np.zeros(self._nnz_jacobian_g)
        self._cached_jacobian_eq = coo_matrix((np.zeros(self._nnz_jacobian_eq),
                                               (self._irows_jac_eq, self._jcols_jac_eq)),
                                               shape=(self._n_eq, self._n_primals))
        self._cached_jacobian_ineq = coo_matrix((np.zeros(self._nnz_jacobian_ineq),
                                                 (self._irows_jac_ineq, self._jcols_jac_ineq)),
                                                shape=(self._n_ineq, self._n_primals))
        self._cached_hessian_lag = coo_matrix((np.zeros(self._nnz_hessian_lag),
                                               (self._irows_hess, self._jcols_hess)),
                                              shape=(self._n_primals, self._n_primals))

        self._invalidate_cache()

    def _invalidate_cache(self):
        self._objective_is_cached = False
        self._grad_objective_is_cached = False
        self._eq_is_cached = False
        self._ineq_is_cached = False
        self._jacobian_eq_is_cached = False
        self._jacobian_ineq_is_cached = False
        self._hessian_lag_is_cached = False

    def _collect_nlp_structure(self):
        """
        Collect characteristics of the NLP from the ASL interface
        """
        # Note: ASL uses "g" which contains all equalities and inequalities
        # in a single vector. These will need to mapped to "eq" and "ineq"
        
        # get the problem dimensions
        self._n_primals = self._asl.get_n_vars()
        self._n_g = self._asl.get_n_constraints()
        self._nnz_jacobian_g = self._asl.get_nnz_jac_g()
        self._nnz_hess_lag_lower = self._asl.get_nnz_hessian_lag()

        # get the initial values for the primals 
        self._init_primals = np.zeros(self._n_primals)
        self._init_duals_g = np.zeros(self._n_g)
        self._asl.get_init_x(self._init_primals)
        self._asl.get_init_multipliers(self._init_duals_g)
        self._init_primals.flags.writeable = False
        self._init_duals_g.flags.writeable = False

        # get the bounds on the primal variables
        self._primals_lb = np.zeros(self._n_primals, dtype=np.double)
        self._primals_ub = np.zeros(self._n_primals, dtype=np.double)
        self._asl.get_x_lower_bounds(self._primals_lb)
        self._asl.get_x_upper_bounds(self._primals_ub)
        self._primals_lb.flags.writeable = False
        self._primals_ub.flags.writeable = False

        # get the bounds on the constraints (equality and
        # inequality are mixed for the asl interface)
        self._g_lb = np.zeros(self._n_g, dtype=np.double)
        self._g_ub = np.zeros(self._n_g, dtype=np.double)
        self._asl.get_g_lower_bounds(self._g_lb)
        self._asl.get_g_upper_bounds(self._g_ub)
        
        # build the  maps for converting from the condensed
        # vectors to the full vectors (e.g., the condensed
        # vector of lower bounds to the full primals vector)
        # Also build the  maps for converting from the full g
        # vector (which includes all equality and inequality
        # constraints) to the equality and inequality
        # constraints on their own.
        self._build_primals_bound_maps()
        self._build_constraint_maps()

        # get the values for the lower and upper bounds on the
        # inequalities (extracted from "g")
        self._ineq_lb = np.compress(self._g_ineq_mask, self._g_lb)
        self._ineq_ub = np.compress(self._g_ineq_mask, self._g_ub)
        self._ineq_lb.flags.writeable = False
        self._ineq_ub.flags.writeable = False

        # get the initial values for the dual variables
        self._init_duals_eq = np.compress(self._g_eq_mask, self._init_duals_g)
        self._init_duals_ineq = np.compress(self._g_ineq_mask, self._init_duals_g)
        self._init_duals_eq.flags.writeable = False
        self._init_duals_eq.flags.writeable = False
        
        # store the rhs of g for equality constraints only
        self._g_rhs = self._g_ub.copy()
        # do not store a rhs for inequality constraints (will use lb, ub)
        self._g_rhs[~self._g_eq_mask] = 0.0
        # change the upper and lower bounds on g
        # to zero for equality constraints
        self._g_lb[self._g_eq_mask] = 0.0
        self._g_ub[self._g_eq_mask] = 0.0
        self._g_lb.flags.writeable = False
        self._g_ub.flags.writeable = False
        
        # set number of equatity and inequality constraints from maps
        self._n_eq = len(self._eq_g_map)
        self._n_ineq = len(self._ineq_g_map)

        # populate jacobian structure
        self._irows_jac_g = np.zeros(self._nnz_jacobian_g, dtype=np.intc)
        self._jcols_jac_g = np.zeros(self._nnz_jacobian_g, dtype=np.intc)
        self._asl.struct_jac_g(self._irows_jac_g, self._jcols_jac_g)
        self._irows_jac_g -= 1
        self._jcols_jac_g -= 1

        self._nz_g_eq_mask = np.isin(self._irows_jac_g, self._eq_g_map)
        self._nz_g_ineq_mask = np.logical_not(self._nz_g_eq_mask)
        self._irows_jac_eq = np.compress(self._nz_g_eq_mask, self._irows_jac_g)
        self._jcols_jac_eq = np.compress(self._nz_g_eq_mask, self._jcols_jac_g)
        self._irows_jac_ineq = np.compress(self._nz_g_ineq_mask, self._irows_jac_g)
        self._jcols_jac_ineq = np.compress(self._nz_g_ineq_mask, self._jcols_jac_g)
        self._nnz_jacobian_eq = len(self._irows_jac_eq)
        self._nnz_jacobian_ineq = len(self._irows_jac_ineq)

        # this is expensive but only done once.
        # Could be vectorized or done from the c-side
        g_eq_map = {self._eq_g_map[i]: i for i in range(self._n_eq)}
        for i, v in enumerate(self._irows_jac_eq):
            self._irows_jac_eq[i] = g_eq_map[v]

        g_ineq_map = {self._ineq_g_map[i]: i for i in range(self._n_ineq)}
        for i, v in enumerate(self._irows_jac_ineq):
            self._irows_jac_ineq[i] = g_ineq_map[v]

        self._irows_jac_eq.flags.writeable = False
        self._jcols_jac_eq.flags.writeable = False
        self._irows_jac_ineq.flags.writeable = False
        self._jcols_jac_ineq.flags.writeable = False
        self._irows_jac_g.flags.writeable = False
        self._jcols_jac_g.flags.writeable = False

        # set nnz for equality and inequality jacobian
        self._nnz_jac_eq = len(self._jcols_jac_eq)
        self._nnz_jac_ineq = len(self._jcols_jac_ineq)

        # populate hessian structure (lower triangular)
        self._irows_hess = np.zeros(self._nnz_hess_lag_lower, dtype=np.intc)
        self._jcols_hess = np.zeros(self._nnz_hess_lag_lower, dtype=np.intc)
        self._asl.struct_hes_lag(self._irows_hess, self._jcols_hess)
        self._irows_hess -= 1
        self._jcols_hess -= 1

        # rework hessian to full matrix (lower and upper)
        diff = self._irows_hess - self._jcols_hess
        self._lower_hess_mask = np.where(diff != 0)
        lower = self._lower_hess_mask
        self._irows_hess = np.concatenate((self._irows_hess, self._jcols_hess[lower]))
        self._jcols_hess = np.concatenate((self._jcols_hess, self._irows_hess[lower]))
        self._nnz_hessian_lag = self._irows_hess.size

        self._irows_hess.flags.writeable = False
        self._jcols_hess.flags.writeable = False

    def _build_primals_bound_maps(self):
        """Creates internal maps and masks for converting from the full vector of
        primal variables to the vector of lower bounds only and upper bounds only
        """
        # sanity check for bounds on the primals
        # TODO: this tolerance should somehow be linked to the algorithm tolerance?
        tolerance_fixed_bounds = 1e-8
        bounds_difference = self._primals_ub - self._primals_lb
        abs_bounds_difference = np.absolute(bounds_difference)
        fixed_vars = np.any(abs_bounds_difference < tolerance_fixed_bounds)
        if fixed_vars:
            print(np.where(abs_bounds_difference<tolerance_fixed_bounds))
            raise RuntimeError("Variables fixed using bounds is not currently supported.")
        
        inconsistent_bounds = np.any(bounds_difference < 0.0)
        if inconsistent_bounds:
            # TODO: improve error message
            raise RuntimeError("Variables found with upper bounds set below the lower bounds.")

        # build lower and upper bound maps for the primals
        self._primals_lb_mask = np.isfinite(self._primals_lb)
        self._lb_primals_map = self._primals_lb_mask.nonzero()[0]
        self._primals_ub_mask = np.isfinite(self._primals_ub)
        self._ub_primals_map = self._primals_ub_mask.nonzero()[0]
        self._primals_lb_mask.flags.writeable = False
        self._lb_primals_map.flags.writeable = False
        self._primals_ub_mask.flags.writeable = False
        self._ub_primals_map.flags.writeable = False

    def _build_constraint_maps(self):
        """Creates internal maps and masks that convert from the full
        vector of constraints (the "g" vector that includes all equality
        and inequality constraints combined) to the vectors that include
        the equality and inequality constraints only.
        """
        # sanity check for bounds on g
        bounds_difference = self._g_ub - self._g_lb
        inconsistent_bounds = np.any(bounds_difference < 0.0)
        if inconsistent_bounds:
            raise RuntimeError("Bounds on inequality constraints found with upper bounds set below the lower bounds.")

        # build maps from g to equality and inequality
        abs_bounds_difference = np.absolute(bounds_difference)
        tolerance_equalities = 1e-8
        self._g_eq_mask = abs_bounds_difference < tolerance_equalities
        self._eq_g_map = self._g_eq_mask.nonzero()[0]
        self._g_ineq_mask = abs_bounds_difference >= tolerance_equalities
        self._ineq_g_map = self._g_ineq_mask.nonzero()[0]
        self._g_eq_mask.flags.writeable = False
        self._eq_g_map.flags.writeable = False
        self._g_ineq_mask.flags.writeable = False
        self._ineq_g_map.flags.writeable = False

        #TODO: Can we simplify this logic?
        g_glb_mask = np.isfinite(self._g_lb) * self._g_ineq_mask + self._g_eq_mask
        lb_g_map = g_glb_mask.nonzero()[0]
        g_gub_mask = np.isfinite(self._g_ub) * self._g_ineq_mask + self._g_eq_mask
        ub_g_map = g_gub_mask.nonzero()[0]

        self._ineq_lb_mask = np.isin(self._ineq_g_map, lb_g_map)
        self._lb_ineq_map = np.where(self._ineq_lb_mask)[0]
        self._ineq_ub_mask = np.isin(self._ineq_g_map, ub_g_map)
        self._ub_ineq_map = np.where(self._ineq_ub_mask)[0]
        self._ineq_lb_mask.flags.writeable = False
        self._lb_ineq_map.flags.writeable = False
        self._ineq_ub_mask.flags.writeable = False
        self._ub_ineq_map.flags.writeable = False


    # overloaded from NLP
    def n_primals(self):
        return self._n_primals

    # overloaded from NLP
    def n_eq_constraints(self):
        return self._n_eq

    # overloaded from NLP
    def n_ineq_constraints(self):
        return self._n_ineq

    # overloaded from NLP
    def nnz_jacobian_eq(self):
        return self._nnz_jacobian_eq

    # overloaded from NLP
    def nnz_jacobian_ineq(self):
        return self._nnz_jacobian_ineq

    # overloaded from NLP
    def nnz_hessian_lag(self):
        return self._nnz_hessian_lag

    # overloaded from NLP
    def primals_lb(self, condensed=False):
        if condensed:
            return self._primals_lb.compress(self._primals_lb_mask)
        return self._primals_lb

    # overloaded from NLP
    def primals_ub(self, condensed=False):
        if condensed:
            return self._primals_ub.compress(self._primals_ub_mask)
        return self._primals_ub
    
    # overloaded from NLP
    def ineq_lb(self, condensed=False):
        if condensed:
            return self._ineq_lb.compress(self._ineq_lb_mask)
        return self._ineq_lb

    # overloaded from NLP
    def ineq_ub(self, condensed=False):
        if condensed:
            return self._ineq_ub.compress(self._ineq_ub_mask)
        return self._ineq_ub

    # overloaded from NLP
    def init_primals(self):
        return self._init_primals

    # overloaded from NLP
    def init_duals_eq(self):
        return self._init_duals_eq

    # overloaded from NLP
    def init_duals_ineq(self):
        return self._init_duals_ineq

    # overloaded from NLP
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
        numpy.ndarray
        """
        if vector_type == 'primals':
            return np.zeros(self._n_primals, dtype=np.double)
        elif vector_type == 'primals_lb_condensed':
            nx_l = len(self._lb_primals_map)
            return np.zeros(nx_l, dtype=np.double)
        elif vector_type == 'primals_ub_condensed':
            nx_u = len(self._ub_primals_map)
            return np.zeros(nx_u, dtype=np.double)
        elif vector_type == 'eq' or vector_type == 'duals_eq':
            return np.zeros(self._n_eq, dtype=np.double)
        elif vector_type == 'ineq' or vector_type == 'duals_ineq':
            return np.zeros(self._n_ineq, dtype=np.double)
        elif vector_type == 'ineq_lb_condensed':
            n_ineq_lb = len(self._lb_ineq_map)
            return np.zeros(n_ineq_lb, dtype=np.double)
        elif vector_type == 'ineq_ub_condensed':
            n_ineq_ub = len(self._ub_ineq_map)
            return np.zeros(n_ineq_ub, dtype=np.double)
        else:
            raise RuntimeError('Called create_new_vector with an unknown vector_type')

    # overloaded from NLP
    def set_primals(self, primals):
        self._invalidate_cache()
        np.copyto(self._primals, primals)

    # overloaded from NLP
    def set_duals_eq(self, duals_eq):
        self._invalidate_cache()
        np.copyto(self._duals_eq, duals_eq)

    # overloaded from NLP
    def set_duals_ineq(self, duals_ineq):
        self._invalidate_cache()
        np.copyto(self._duals_ineq, duals_ineq)

    def _evaluate_objective_and_cache_if_necessary(self):
        if not self._objective_is_cached:
            self._cached_objective = self._asl.eval_f(self._primals)
            self._objective_is_cached = True

    # overloaded from NLP
    def evaluate_objective(self):
        self._evaluate_objective_and_cache_if_necessary()
        return self._cached_objective

    # overloaded from NLP
    def evaluate_grad_objective(self, out=None):
        if not self._grad_objective_is_cached:
            self._asl.eval_deriv_f(self._primals, self._cached_grad_objective)
            self._grad_objective_is_cached = True

        if out is not None:
            if not isinstance(out, np.ndarray) or out.size != self._n_primals:
                raise RuntimeError('Called evaluate_grad_objective with an invalid "out" argument - should take an ndarray of size {}'.format(self._n_primals))
            np.copyto(out, self._cached_grad_objective)
        else:
            return self._cached_grad_objective.copy()

    def _evaluate_constraints_and_cache_if_necessary(self):
        assert(self._eq_is_cached == self._ineq_is_cached)
        if not self._eq_is_cached or not self._ineq_is_cached:
            self._asl.eval_g(self._primals, self._cached_g)
            self._cached_g -= self._g_rhs
            self._cached_g.compress(self._g_eq_mask, out=self._cached_eq)
            self._cached_g.compress(self._g_ineq_mask, out=self._cached_ineq)
            self._eq_is_cached = True
            self._ineq_is_cached = True
    
    # overloaded from NLP
    def evaluate_constraints_eq(self, out=None):
        self._evaluate_constraints_and_cache_if_necessary()

        if out is not None:
            if not isinstance(out, np.ndarray) or out.size != self._n_eq:
                raise RuntimeError('Called evaluate_constraints_eq with an invalid'
                                   ' "out" argument - should take an ndarray of '
                                   'size {}'.format(self._n_eq))
            np.copyto(out, self._cached_eq)
        else:
            return self._cached_eq.copy()

    # overloaded from NLP
    def evaluate_constraints_ineq(self, out=None):
        self._evaluate_constraints_and_cache_if_necessary()

        if out is not None:
            if not isinstance(out, np.ndarray) or out.size != self._n_ineq:
                raise RuntimeError('Called evaluate_constraints_ineq with an invalid'
                                   ' "out" argument - should take an ndarray of '
                                   'size {}'.format(self._n_ineq))
            np.copyto(out, self._cached_ineq)
        else:
            return self._cached_ineq.copy()

    def _evaluate_jacobians_and_cache_if_necessary(self):
        assert self._jacobian_eq_is_cached == self._jacobian_ineq_is_cached
        if not self._jacobian_eq_is_cached or not self._jacobian_ineq_is_cached:
            self._asl.eval_jac_g(self._primals, self._cached_jacobian_g_data)
            self._cached_jacobian_eq.data = \
                self._cached_jacobian_g_data.compress(self._nz_g_eq_mask)
            self._cached_jacobian_ineq.data = \
                self._cached_jacobian_g_data.compress(self._nz_g_ineq_mask)
            self._jacobian_eq_is_cached = True
            self._jacobian_ineq_is_cached = True
        
    # overloaded from NLP
    def evaluate_jacobian_eq(self, out=None):
        self._evaluate_jacobians_and_cache_if_necessary()

        if out is not None:
            if not isinstance(out, coo_matrix) or out.shape[0] != self._n_eq or out.shape[1] != self._n_primals:
                raise RuntimeError('evaluate_jacobian_eq called with an "out" argument'
                                   ' that is invalid. This should be a coo_matrix with'
                                   ' shape=({},{})'.format(self._n_eq, self._n_primals))
            np.copyto(out.data, self._cached_jacobian_eq.data)
        else:
            return self._cached_jacobian_eq.copy()

    # overloaded from NLP
    def evaluate_jacobian_ineq(self, out=None):
        self.evaluate_jacobians_and_cache_if_necessary()

        if out is not None:
            if not isinstance(out, coo_matrix) or out.shape[0] != self._n_ineq or out.shape[1] != self._n_primals:
                raise RuntimeError('evaluate_jacobian_ineq called with an "out" argument'
                                   ' that is invalid. This should be a coo_matrix with'
                                   ' shape=({},{})'.format(self._n_ineq, self._n_primals))
            np.copyto(out.data, self._cached_jacobian_ineq.data)
        else:
            return self._cached_jacobian_ineq.copy()

    def evaluate_hessian_lag(self, out=None):
        if not self._hessian_is_cached:
            # evaluating the hessian requires that we have first
            # evaluated the objective and the constraints
            self._evaluate_objective_and_cache_if_necessary()
            self._evaluate_constraints_and_cache_if_necessary()

            # TODO: support objective scaling factor
            obj_factor = 1.0

            # get the "g" duals from the eq and ineq duals
            self._duals_g[self._g_eq_mask] = self._duals_eq
            self._duals_g[self._g_ineq_mask] = self._duals_ineq
            
            # get the hessian
            data = np.zeros(self._nns_hess_lag_lower, np.double)
            self._asl.eval_hes_lag(self._primals, self._duals_g,
                                   data, obj_factor=obj_factor)
            values = np.concatenate((data, data[self._lower_hess_mask]))
            #TODO: find out why this is done
            values += 1e-16 # this is to deal with scipy bug temporarily
            self._cached_hessian_lag.data = values

        if out is not None:
            if not isinstance(out, coo_matrix) or out.shape[0] != self._n_primals or \
               out.shape[1] != self._n_primals or out.nnz != self._nnz_hessian_lag:
                raise RuntimeError('evaluate_hessian_lag called with an "out" argument'
                                   ' that is invalid. This should be a coo_matrix with'
                                   ' shape=({},{})'.format(self._n_primals, self._n_primals))
            out.data = self._cached_hessian_lag.data.copy()
        else:
            return self._cached_hessian_lag.copy()

    # overloaded from NLP
    def primals_lb_condensing_matrix(self):
        col = self._lb_primals_map
        nnz = len(self._lb_primals_map)
        row = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return csr_matrix((data, (row, col)), shape=(self.nx, nnz)).toocoo()

    # overloaded from NLP
    def primals_ub_condensing_matrix(self):
        col = self._ub_primals_map
        nnz = len(self._ub_primals_map)
        row = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return csr_matrix((data, (row, col)), shape=(self.nx, nnz)).toocoo()

    # overloaded from NLP
    def ineq_lb_condensing_matrix(self):
        col = self._lb_ineq_map
        nnz = len(self._lb_ineq_map)
        row = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return csr_matrix((data, (row, col)), shape=(self.nx, nnz)).toocoo()

    # overloaded from NLP
    def ineq_ub_condensing_matrix(self):
        col = self._ub_ineq_map
        nnz = len(self._ub_ineq_map)
        row = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return csr_matrix((data, (row, col)), shape=(self.nx, nnz)).toocoo()

    def report_solver_status(self, status_code, status_message, primals, duals_eq, duals_ineq):
        # get the "g" duals from the eq and ineq duals
        duals_g = np.zeros(len(self._duals_g))
        duals_g[self._g_eq_mask] = duals_eq
        duals_g[self._g_ineq_mask] = duals_ineq

        self._asl.finalize_solution(status_code, status_message, primals, duals_g)


